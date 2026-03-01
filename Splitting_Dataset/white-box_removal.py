import os
import argparse
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import wavmark
from wavmark.utils import wm_add_util
import yaml
from Timbre_10.model.conv2_mel_modules import Decoder
import silentcipher
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    parser.add_argument("--length", type=int, default=5*16000, help="Length of the audio samples")

    parser.add_argument("--gpu", type=int, default=7, help="GPU device index to use")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input audio files to be attacked")
    
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations for the attack")
    parser.add_argument("--pert_boundary", type=float, default=0.001, help="Perturbation boundary for the attack")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the attack")
    parser.add_argument("--rescale_snr", type=float, default=60, help="rescaled SNR after applying the perturbation")
    
    parser.add_argument("--whitebox_folder", type=str, default="whitebox_debug", help="Folder to save the whitebox attack results")
    
    parser.add_argument("--tau", type=float, default=0.1, help="Threshold for the detector")

    parser.add_argument("--attack_bitstring", action="store_true", default=False, help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='audioseal', choices=['audioseal','wavmark', 'timbre', 'silentcipher', 'echo', 'dsss', 'phase','patchwork', 'qim'], help="Model to be attacked")
    

    print("Arguments: ", parser.parse_args())
    return parser.parse_args()
    

def api_visqol():
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api

class SilentCipherWrapper(nn.Module):
    def __init__(self, silentcipher_model):
        super(SilentCipherWrapper, self).__init__()
        self.model = silentcipher_model

    def encode_wav(self, *args, **kwargs):
        return self.model.encode_wav(*args, **kwargs)

    def decode_wav(self, *args, **kwargs):
        return self.model.decode_wav(*args, **kwargs)
    



class EchoHidingWrapper(nn.Module):
    def __init__(self, msg_length=16, L=4096, d0=150, d1=200):
        super().__init__()
        self.msg_length = msg_length
        self.L = L  # Frame length
        self.d0 = d0
        self.d1 = d1
        self.alpha = 0.5  # Must match embedding parameter
        self.K = 256  # Smoothing window size


    def forward(self, x):
        """
        Differentiable Echo Hiding decoder using PyTorch
        x: [1, T] tensor
        """
        # Ensure proper shape
        if x.dim() == 3:
            x = x.squeeze(1)  # [1, T]
        if x.dim() == 2:
            x = x.squeeze(0)  # [T]
        
        T = x.shape[-1]
        N = T // self.L  # Number of frames
        
        if N == 0:
            # Return tensor with gradients enabled
            return torch.zeros(self.msg_length, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)

        # Reshape into frames
        frames_data = x[:N * self.L]
        frames = frames_data.view(N, self.L).t()  # [L, N]
        
        detected_bits = []
        
        for k in range(min(N, self.msg_length)):
            frame = frames[:, k]  # [L]
            
            # Compute magnitude spectrum
            spectrum = torch.abs(torch.fft.fft(frame))
            
            # Add small epsilon to avoid log(0)
            log_spectrum = torch.log(spectrum + 1e-12)
            
            # Compute real cepstrum
            cepstrum = torch.fft.ifft(log_spectrum).real  # [L]
            
            # Extract peaks at delay locations
            peak0 = torch.abs(cepstrum[self.d0])
            peak1 = torch.abs(cepstrum[self.d1])
            
            # Soft decision for differentiability
            diff = peak1 - peak0
            prob = torch.sigmoid(diff * 10)  # Sharpen decision
            detected_bits.append(prob)
        
        
        
        # Use torch.stack with proper gradient handling
        result = torch.stack(detected_bits[:self.msg_length])
        return result
    
class DSSSWrapper(nn.Module):
    def __init__(self, msg_length=16, L_min=4096):
        super().__init__()
        self.msg_length = msg_length
        self.L_min = L_min  # Minimum segment length


    def forward(self, x):
        """
        Differentiable DSSS decoder using PyTorch
        x: [1, T] tensor
        """
        # Ensure proper shape
        if x.dim() == 3:
            x = x.squeeze(1)  # [1, T]
        if x.dim() == 2:
            x = x.squeeze(0)  # [T]
        
        s_len = x.shape[-1]
        L2 = s_len // self.msg_length
        L = max(self.L_min, L2)
        nframe = s_len // L
        N = min(nframe, self.msg_length)
        
        if N == 0:
            return torch.zeros(self.msg_length, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)

        # Reshape signal into segments (column-major like MATLAB)
        xsig_data = x[:N * L]
        xsig = xsig_data.view(N, L).t()  # [L, N] - transpose for column-major
        
        # Generate reference signal
        r = torch.ones(L, device=x.device, dtype=x.dtype)
        
        # Correlation detection with soft decision
        detected_probs = []
        for k in range(N):
            # Normalize correlation for differentiability
            correlation = torch.sum(xsig[:, k] * r) / L
            
            # Soft decision: map correlation to probability
            # Negative correlation -> bit 0, Positive -> bit 1
            prob = torch.sigmoid(correlation * 10)  # Sharpen decision boundary
            detected_probs.append(prob)
        
        
        
        result = torch.stack(detected_probs[:self.msg_length])
        return result
    
class PhaseCodingWrapper(nn.Module):
    def __init__(self, msg_length=16, L=1024):
        super().__init__()
        self.msg_length = msg_length
        self.L = L  # Frame length for FFT



    def forward(self, x):
        """
        Differentiable Phase Coding decoder - EXACT MATCH to NumPy version
        """
        # Ensure proper shape (match NumPy behavior)
        if x.dim() == 3:
            x = x.squeeze(1)  # [1, T] -> [T]
        if x.dim() == 2:
            x = x.squeeze(0)  # [1, T] -> [T]
        
        # Get first frame (exact same as NumPy)
        if x.shape[-1] < self.L:
            # Pad with zeros if signal too short (match NumPy behavior)
            padding = torch.zeros(self.L - x.shape[-1], device=x.device, dtype=x.dtype)
            x_frame = torch.cat([x, padding])
        else:
            x_frame = x[:self.L]
        
        # Compute FFT and get phase angles (exact same as NumPy)
        X = torch.fft.fft(x_frame)
        Phi = torch.angle(X)  # Phase angles
        
        # Calculate middle point (exact same as NumPy)
        mid = self.L // 2
        
        # Extract bits from phase values with differentiable decision
        detected_probs = []
        for k in range(self.msg_length):
            # Exact same indexing as NumPy: mid - L_msg + k
            idx = mid - self.msg_length + k
            
            # Bounds check (match NumPy behavior)
            if idx >= 0 and idx < len(Phi):
                phase_val = Phi[idx]
                
                # EXACT SAME DECISION LOGIC as NumPy:
                # if phase_val > 0: bit = '0' else: bit = '1'
                
                # Convert to differentiable probability:
                # phase_val > 0 -> prob close to 1 (represents bit '0')
                # phase_val <= 0 -> prob close to 0 (represents bit '1')
                prob = torch.sigmoid(phase_val * 10)  # Sharp sigmoid
            else:
                # Out of bounds - default to 0.5 (uncertain)
                prob = torch.tensor(0.5, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
            
            detected_probs.append(prob)
        
        # Stack all probabilities into single tensor
        result = torch.stack(detected_probs)
        return result



class PatchworkWrapper(nn.Module):
    def __init__(self, msg_length=16, sr=16000, fs=3000, fe=7000, k1=0.195, k2=0.08):
        super().__init__()
        self.msg_length = msg_length
        self.sr = sr
        self.fs = fs
        self.fe = fe
        self.k1 = k1
        self.k2 = k2


    def forward(self, x):
        """
        Differentiable Patchwork decoder - EXACT MATCH to original NumPy version
        """
        # Ensure proper shape
        if x.dim() == 3:
            x = x.squeeze(1)  # [1, T]
        if x.dim() == 2:
            x = x.squeeze(0)  # [T]
        
        L = x.shape[-1]
        
        # Calculate indices exactly as original
        si = int(self.fs / (self.sr / L))
        ei = int(self.fe / (self.sr / L))
        
        # Apply DCT - using approximation or torch_dct
        try:
            import torch_dct
            X = torch_dct.dct(x, type=2, norm='ortho')
        except ImportError:
            # Fallback DCT approximation
            X = self.dct_type2_approx(x)
        
        # Extract frequency range exactly as original
        if ei + 1 <= len(X):
            Xs = X[si:(ei + 1)]
        else:
            # Pad with zeros to match original behavior
            pad_needed = ei + 1 - len(X) + si
            if pad_needed > 0:
                padding = torch.zeros(pad_needed, device=x.device, dtype=x.dtype)
                Xs = torch.cat([X[si:], padding])
            else:
                Xs = X[si:ei+1]
        
        Ls = len(Xs)
        
        # Adjust length exactly as original
        if Ls % (self.msg_length * 2) != 0:
            Ls = Ls - (Ls % (self.msg_length * 2))
            Xs = Xs[:Ls]
        
        if Ls <= 0:
            return torch.zeros(self.msg_length, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
        
        # THIS IS THE CRITICAL PART - EXACT MATCH TO ORIGINAL:
        # Xsp = np.dstack((Xs[:Ls // 2], Xs[:(Ls // 2 - 1):-1])).flatten()
        Ls_half = Ls // 2
        
        # First half: Xs[:Ls // 2]
        first_half = Xs[:Ls_half]
        # Second half: Xs[:(Ls // 2 - 1):-1] (reversed second half)
        second_half = torch.flip(Xs[Ls_half:], dims=[0])
        
        # Stack them as pairs and flatten - EXACT MATCH to np.dstack(...).flatten()
        # np.dstack creates shape (Ls_half, 2) then flattens
        Xsp_pairs = torch.stack([first_half, second_half], dim=1)  # [Ls_half, 2]
        Xsp = Xsp_pairs.flatten()  # Flatten to 1D - exact match
        
        # Split into segments exactly as original
        segment_length = len(Xsp) // (self.msg_length * 2)
        if segment_length == 0:
            return torch.zeros(self.msg_length, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
        
        segments = Xsp.split(segment_length)
        
        # Extract watermark bits exactly as original
        watermark_probs = []
        for i in range(0, min(len(segments), self.msg_length * 2), 2):
            if i + 1 >= len(segments):
                break
                
            j = i // 2 + 1
            
            # Calculate means exactly as original
            m1j = torch.mean(torch.abs(segments[i]))
            m2j = torch.mean(torch.abs(segments[i + 1]))
            
            dj = m1j - m2j
            
            # EXACT SAME DECISION LOGIC as original:
            # if dj >= 0: append(0) else: append(1)
            
            # But make it differentiable:
            # dj >= 0 -> prob close to 0 (bit 0)
            # dj < 0 -> prob close to 1 (bit 1)
            prob = torch.sigmoid(-dj * 10)  # Sharp decision
            watermark_probs.append(prob)
        
        # Pad if needed (match original behavior)
        while len(watermark_probs) < self.msg_length:
            # Create tensor with proper gradient tracking
            zero_prob = torch.tensor(0.5, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
            watermark_probs.append(zero_prob)
        
        # Return exactly msg_length bits
        return torch.stack(watermark_probs[:self.msg_length])
    
    def dct_type2_approx(self, x):
        """
        DCT Type II approximation using FFT
        """
        N = len(x)
        if N == 0:
            return torch.zeros(0, device=x.device, dtype=x.dtype)
        
        # Pad to even length
        if N % 2 == 1:
            x = torch.cat([x, torch.zeros(1, device=x.device, dtype=x.dtype)])
            N += 1
            if N == 1:
                return x[:1]
        
        # Create extended signal
        x_ext = torch.cat([x[::2], x[1::2].flip(dims=[0])])
        
        # Compute FFT
        X_fft = torch.fft.fft(x_ext)
        
        # Extract DCT coefficients
        k = torch.arange(N, device=x.device, dtype=x.dtype)
        dct_coeffs = 2 * torch.exp(-1j * torch.pi * k / (2 * N)) * X_fft[:N]
        dct_real = torch.real(dct_coeffs)
        
        # Normalization
        dct_real = dct_real * torch.sqrt(torch.tensor(2.0 / N, device=x.device, dtype=x.dtype))
        if len(dct_real) > 0:
            dct_real[0] = dct_real[0] * torch.sqrt(torch.tensor(0.5, device=x.device, dtype=x.dtype))
        
        return dct_real[:len(x)]

class QIMDetectorWrapper(nn.Module):
    def __init__(self, delta=0.01, msg_length=16, step=100):
        super().__init__()
        self.delta = delta
        self.msg_length = msg_length
        self.step = step




    def forward(self, x):
        """
        Differentiable QIM decoder - EXACT MATCH with proper gradients
        """
        # Ensure proper shape
        if x.dim() == 3:
            x = x.squeeze(1)  # [1, T]
        if x.dim() == 2:
            x = x.squeeze(0)  # [T]
        
        detected_probs = []
        
        # Extract samples at the same intervals
        sample_indices = list(range(0, len(x), self.step))[:self.msg_length]
        
        for i in sample_indices:
            if len(detected_probs) >= self.msg_length:
                break
                
            sample = x[i]
            
            # EXACT QIM detection logic from original
            sample_div_delta = sample / self.delta
            
            # Differentiable round
            rounded = torch.round(sample_div_delta)
            
            # Calculate quantization points exactly as original
            # z0 = round(sample/delta) * delta + (-1)**(0+1) * delta/4 = round*delta - delta/4
            # z1 = round(sample/delta) * delta + (-1)**(1+1) * delta/4 = round*delta + delta/4
            z0 = rounded * self.delta - self.delta/4.0
            z1 = rounded * self.delta + self.delta/4.0
            
            # Calculate distances
            d0 = torch.abs(sample - z0)
            d1 = torch.abs(sample - z1)
            
            # EXACT SAME DECISION LOGIC as original:
            # if d0 < d1: append(0) else: append(1)
            
            # Convert to differentiable probability using the decision difference
            # Original logic: d0 < d1 -> bit 0, d0 >= d1 -> bit 1
            # distance_diff = d1 - d0: positive when d0 < d1
            # We want: d0 < d1 -> prob close to 0 (bit 0), d0 >= d1 -> prob close to 1 (bit 1)
            distance_diff = d1 - d0  # positive when d0 < d1
            prob = torch.sigmoid(-distance_diff * 10)  # Negative for correct mapping
            detected_probs.append(prob)
        
        # Pad if needed with proper gradient tracking
        while len(detected_probs) < self.msg_length:
            # Create tensor that inherits gradient properties from input
            zero_prob = torch.tensor(0.5, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
            detected_probs.append(zero_prob)
        
        # Stack all probabilities (this preserves gradients)
        if detected_probs:
            result = torch.stack(detected_probs[:self.msg_length])
        else:
            result = torch.zeros(self.msg_length, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)
        
        return result

class WatermarkDetectorWrapper():
    def __init__(self, model, message, on_bitstring, model_type, threshold, device):
        

        self.model = model
        self._device = device
        self.message = message.to(self._device)
        self.on_bitstring = on_bitstring
        if model_type == 'echo':
            self.echo = EchoHidingWrapper(msg_length=len(message)).to(device)
        if model_type == 'dsss':
            self.dsss = DSSSWrapper(msg_length=len(message)).to(device)
        if model_type == 'phase':
            self.phase = PhaseCodingWrapper(msg_length=len(message)).to(device)
        if model_type == 'patchwork':
            self.patchwork = PatchworkWrapper(msg_length=len(message)).to(device)
        if model_type == 'qim':
            self.qim = QIMDetectorWrapper(delta=0.01, msg_length=len(message)).to(device)


        elif model is not None:  
            self.model.to(self._device)
        self.model_type = model_type
        self.threshold = threshold
        if model_type == 'timbre':
            self.bwacc = self.bwacc_timbre
            self.get_loss = self.loss_timbre
        elif model_type == 'audioseal':
            self.bwacc = self.bwacc_audioseal
            self.get_loss = self.loss_audioseal
        elif model_type == 'wavmark':
            wavmark_start_bit = wm_add_util.fix_pattern[0:16]  # 16 bits of watermark
            start_bit = torch.tensor(wavmark_start_bit, dtype=torch.float32)
            self.reversed_start_bit = 1 - start_bit
            self.reversed_start_bit = self.reversed_start_bit.repeat(20, 1)          
            self.reversed_start_bit = self.reversed_start_bit.to(self._device)
            self.total_detect_points = torch.arange(0, 800 * 80, 800)
            self.bwacc = self.bwacc_wavmark
            self.get_loss = self.loss_wavmark
        elif model_type == 'silentcipher':
            self.bwacc = self.bwacc_silentcipher
            self.get_loss = self.loss_silentcipher
        elif model_type == 'echo':
            self.bwacc = self.bwacc_echo_hiding
            self.get_loss = self.loss_echo_hiding
        elif model_type == 'dsss':
            self.bwacc = self.bwacc_dsss
            self.get_loss = self.loss_dsss
        elif model_type  == 'phase':
            self.bwacc = self.bwacc_phase
            self.get_loss = self.loss_phase
        elif model_type == 'patchwork':
            self.bwacc = self.bwacc_patch
            self.get_loss = self.loss_patch
        elif model_type == 'qim':
            self.bwacc = self.bwacc_qim
            self.get_loss = self.loss_qim

    def loss_qim(self, signal):
        decoded_bits = self.qim(signal)
        target_bits = (1-self.message).float()
        decoded_bits = torch.clamp(decoded_bits, 1e-7, 1-1e-7)
        loss_fn = nn.BCELoss()
        loss = loss_fn(decoded_bits, target_bits)
        return loss

    def loss_patch(self, signal):
        decoded_bits = self.patchwork(signal)
        target_bits = (1-self.message).float()
        decoded_bits = torch.clamp(decoded_bits, 1e-7, 1-1e-7)
        loss_fn = nn.BCELoss()
        loss = loss_fn(decoded_bits, target_bits)
        return loss

    def loss_phase(self, signal):
        decoded_bits = self.phase(signal)
        target_bits = (1-self.message).float()
        decoded_bits = torch.clamp(decoded_bits, 1e-7, 1-1e-7)
        loss_fn = nn.BCELoss()
        loss = loss_fn(decoded_bits, target_bits)
        return loss

    def loss_dsss(self, signal):

        decoded_bits = self.dsss(signal)
        target_bits = (1-self.message).float()
        decoded_bits = torch.clamp(decoded_bits, 1e-7, 1-1e-7)
        loss_fn = nn.BCELoss()
        loss = loss_fn(decoded_bits, target_bits)
        return loss

    def loss_echo_hiding(self, signal):
        # signal should be [1, T] or [B, 1, T]
        decoded_bits = self.echo(signal)  # This should return differentiable tensor
        
        
        target_bits = (1 - self.message).float()
        decoded_bits = torch.clamp(decoded_bits, 1e-7, 1 - 1e-7)
        loss_fn = nn.BCELoss()
        loss = loss_fn(decoded_bits, target_bits)
        
        
        return loss

    def loss_silentcipher(self, signal):
        signal_np = signal.squeeze(0).detach().cpu().numpy()
        result = self.model.decode_wav(signal_np, 16000, phase_shift_decoding=False)
        
        # Target: encourage decoding of wrong message
        reversed_message = 1 - self.message
        
        if result['status'] and result['messages'][0] is not None:
            # Get decoded message bits
            binary_sequence = ''.join(format(byte, '08b') for byte in result['messages'][0][:2])
            first_16_bits = binary_sequence[:16]
            decoded_bits = torch.tensor(list(map(int, first_16_bits)), dtype=torch.float32, device=self._device)
            
            # Cross-entropy loss to encourage wrong decoding
            # We want to maximize distance from correct message, so we use reversed message as target
            loss_fn = nn.BCELoss()
            target_bits = reversed_message.float()
            
            # Add small epsilon to avoid log(0)
            decoded_bits = torch.clamp(decoded_bits, 1e-7, 1 - 1e-7)
            target_bits = torch.clamp(target_bits, 1e-7, 1 - 1e-7)
            loss = loss_fn(decoded_bits, target_bits)
            return loss
        else:
            return torch.tensor(0.1, device=self._device, requires_grad=True)
        



    def loss_audioseal(self, signal):
        results, messages = self.model(signal)
        reversed_message = 1 - self.message
        loss = nn.CrossEntropyLoss()
        cross_entropy_loss = loss(messages.squeeze(), reversed_message)
        class_1_probs = results[:, 1, :]
        penalty = torch.relu(class_1_probs)
        total_penalty = torch.sum(penalty)
        return total_penalty + cross_entropy_loss
    
    def loss_timbre(self, signal):
        payload = self.model.test_forward(signal) # signal: [1,1,80000]
        message = (1 - self.message) * 2 - 1
        payload = payload.to(message.device)
        loss = nn.CrossEntropyLoss()
        cross_entropy_loss = loss(payload.squeeze(), message)
        return cross_entropy_loss

    def loss_wavmark(self, signal):
        signal = signal.squeeze()
        select_indices = torch.randint(0, 80, (20,))
        detect_points = self.total_detect_points[select_indices]
        slices = torch.stack([signal[..., p:p + 16000] for p in detect_points]).to(self._device)
        batch_messages = self.model.decode(slices)
        decoded_start_bits = batch_messages[:, 0:16]
        decoded_messages = batch_messages[:, 16:]
        loss_fn = nn.BCEWithLogitsLoss()
        start_bit_loss = loss_fn(decoded_start_bits, self.reversed_start_bit)
        reversed_msg = 1 - self.message
        msg_loss = loss_fn(decoded_messages, reversed_msg.repeat(20, 1))
        return start_bit_loss + msg_loss 
    
    def bwacc_qim(self, signal):
        decoded_msg = self.qim(signal)
        decoded_bits = (decoded_msg > 0.5).float()
        bitacc = 1 - torch.sum(torch.abs(decoded_bits - self.message)) / self.message.shape[0]
        print(decoded_bits, bitacc)
        return bitacc
    
    def bwacc_patch(self, signal):
        decoded_msg = self.patchwork(signal)
        decoded_bits = (decoded_msg > 0.5).float()
        bitacc = 1 - torch.sum(torch.abs(decoded_bits - self.message)) / self.message.shape[0]
        print(decoded_bits, bitacc)
        return bitacc
    
    def bwacc_phase(self, signal):
        decoded_msg = self.phase(signal)
        decoded_bits = (decoded_msg > 0.5).float()
        bitacc = 1 - torch.sum(torch.abs(decoded_bits - self.message)) / self.message.shape[0]
        print(decoded_bits, bitacc)
        return bitacc

    
    def bwacc_dsss(self, signal):

        decoded_msg = self.dsss(signal)
        decoded_bits = (decoded_msg > 0.5).float()
        bitacc = 1 - torch.sum(torch.abs(decoded_bits - self.message)) / self.message.shape[0]
        print(decoded_bits, bitacc)
        return bitacc
    
    def bwacc_echo_hiding(self, signal):
        
        decoded_msg = self.echo(signal)
        
        decoded_bits = (decoded_msg > 0.5).float()
        
        bitacc = 1 - torch.sum(torch.abs(decoded_bits - self.message)) / self.message.shape[0]
        print(decoded_bits, bitacc)
        return bitacc
    
    def bwacc_silentcipher(self, signal):
        
        signal_np = signal.squeeze(0).cpu().numpy()
        result = self.model.decode_wav(signal_np, 16000, phase_shift_decoding=False)
        print(result)
        if self.on_bitstring:
            if result['status'] != True or result['messages'][0] is None:
                return torch.tensor(0.0, device=self._device)
            
            # Convert byte message to bits
            binary_sequence = ''.join(format(byte, '08b') for byte in result['messages'][0][:2])
            
            # Extract the first 16 bits
            first_16_bits = binary_sequence[:16]
            msg_bits = torch.tensor(list(map(int, first_16_bits)), dtype=torch.float32, device=self._device)
            
            bit_acc = 1 - torch.sum(torch.abs(msg_bits - self.message)) / self.message.shape[0]
            return bit_acc
        else:
            confidence = result['confidences'][0] if result['status'] == True else 0
            return torch.tensor(confidence, device=self._device)
    

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal)
        if self.on_bitstring:
            if msg_decoded is None:
                return torch.zeros(1)
            else: 
                bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / self.message.numel()
                return bitacc
        else:
            return result
        
    def bwacc_wavmark(self, signal):
        signal = signal.squeeze().detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)

        if payload is None:
            return 0
        else: 
            payload = torch.tensor(payload).to(self.message.device)
            bitacc = 1 - torch.sum(torch.abs(self.message - payload)) / self.message.numel()
            return bitacc.item()

    def bwacc_timbre(self, signal):  #signal is tensor on gpu
        payload = self.model.test_forward(signal) # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return bitacc


def get_scale_factor(signal, noise, required_SNR):
    snr = 10*torch.log10(torch.mean(signal**2)/torch.mean(noise**2))
    if snr < required_SNR:
        scale_factor = 10 ** ((required_SNR - snr) / 10)
    else: 
        scale_factor = 1
    return scale_factor


def whitebox_attack(detector, watermarked_signal, args):
    start_time = time.time()
    bwacc = detector.bwacc(watermarked_signal)
    best_bwacc = bwacc
    best_adv_signal = watermarked_signal
    # Initialize tensor_pert
    tensor_pert = torch.zeros_like(watermarked_signal, requires_grad=True)
    # Freeze detector and watermarked_signal
    watermarked_signal.requires_grad = False
    # Define optimizer
    optimizer = optim.Adam([tensor_pert], lr=args.lr)
    # Projected Gradient Descent
    for _ in range(args.iter):
        if detector.model is not None:
            detector.model.train()
        optimizer.zero_grad()
        watermarked_signal_with_noise = watermarked_signal + tensor_pert 
        loss = detector.get_loss(watermarked_signal_with_noise)
        # Backpropagation
        loss.backward()
        optimizer.step()
        scale_factor = get_scale_factor(watermarked_signal, tensor_pert, args.rescale_snr)
        if scale_factor > 1:
            tensor_pert.data /= scale_factor
        # Evaluation
        if detector.model is not None:
            detector.model.eval()
        with torch.no_grad():
            watermarked_signal_with_noise = watermarked_signal + tensor_pert 
            bwacc = detector.bwacc(watermarked_signal_with_noise)
            snr = 10*torch.log10(torch.mean(watermarked_signal**2)/torch.mean(tensor_pert**2))
            if bwacc < best_bwacc:
                best_bwacc = bwacc
                best_adv_signal = watermarked_signal_with_noise
            if best_bwacc <= args.tau:
                break
    if best_bwacc > args.tau:
        print(f'Attack failed, the best bwacc is {best_bwacc}')
    print(f'Attack time: {time.time() - start_time}')
    return best_adv_signal


def decode_audio_files_perturb_whitebox(model, args, device):
    #watermarked_files = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
    #progress_bar = tqdm(enumerate(watermarked_files), desc="Decoding Watermarks under whitebox attack")
    save_path = os.path.join(args.whitebox_folder, "whitebox_removal")
    os.makedirs(save_path, exist_ok=True)
    watermarked_file = os.path.basename(args.input_dir) 
    #visqol = api_visqol()
 
    idx = '_'.join(watermarked_file.split('_')[2:]) # idx_bitstring_snr
    waveform, sample_rate = torchaudio.load(args.input_dir)
    '''waveform.shape = [1, 80000]'''
    waveform = waveform.to(device=device)
    waveform = waveform.unsqueeze(0)

    original_payload_str = watermarked_file.split('_')[1]
    original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.float32, device=device)

    detector = WatermarkDetectorWrapper(model, original_payload, args.attack_bitstring, args.model, args.tau, device)
    adv_signal = whitebox_attack(detector, waveform, args)

    '''save to log file'''
    #filename=os.path.join(save_path, f'whitebox.csv')
    #log = open(filename, 'a' if os.path.exists(filename) else 'w')
    #log.write('idx, query, acc, snr, visqol\n')
    acc = detector.bwacc(adv_signal)
    snr = 10*torch.log10(torch.sum(waveform**2)/torch.sum((adv_signal - waveform)**2))
    #visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
    print(f'idx: {idx}, query: {args.iter}, acc: {acc:.3f}, snr: {snr:.1f}')
    #log.write(f'{idx}, {args.iter}, {acc}, {snr}, {visqol_score}\n')
    
    torchaudio.save(os.path.join(save_path, watermarked_file),
        adv_signal.squeeze(0).detach().cpu(), sample_rate)
             
def main():
    args = parse_arguments()

    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'timbre':
        process_config = yaml.load(open("config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("config/model.yaml", "r"), Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        msg_length = 10
        detector = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load('results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar')
        detector.load_state_dict(checkpoint['decoder'], strict=False)
        detector.eval()
        model = detector
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'silentcipher':
        sc_model = silentcipher.get_model(model_type='44.1k', device=device)
        model = SilentCipherWrapper(sc_model)  # Wrap the SilentCipher model
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'echo':
        model = None  # The wrapper handles the detection
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'dsss':
        model = None  # The wrapper handles the detection
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'phase':
        model = None  # The wrapper handles the detection
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'patchwork':
        model = None  # The wrapper handles the detection
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'qim':
        model = None  # The wrapper handles the detection
        #output_dir = 'path to your watermarked dataset'
        #os.makedirs(output_dir, exist_ok=True)
    



    decode_audio_files_perturb_whitebox(model, args, device)
if __name__ == "__main__":
    main()
