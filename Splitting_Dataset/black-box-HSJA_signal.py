import os
import argparse
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import wavmark
import silentcipher
import yaml
from Timbre_10.model.conv2_mel_modules import Decoder
from tqdm import tqdm
import fnmatch
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from hop_skip_jump import HopSkipJump
import torch.nn as nn
from scipy.fftpack import dct, idct


def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    # parser.add_argument("--encode", default=True, help="Run the encoding process before decoding")

    parser.add_argument("--length", type=int, default=5 * 16000, help="Length of the audio samples")

    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input audio files to be attacked")


    parser.add_argument("--query_budget", type=int, default=10000, help="Query budget for the attack")
    parser.add_argument("--blackbox_folder", type=str, default="HSJ_signal",
                        help="Folder to save the blackbox attack results")

    parser.add_argument("--max_iter", type=int, default=1, help="Maximum number of iterations for the attack")
    parser.add_argument("--max_eval", type=int, default=1000,
                        help="Maximum number of evaluations for estimating gradient")
    parser.add_argument("--init_eval", type=int, default=100,
                        help="Initial number of evaluations for estimating gradient")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the attack")
    parser.add_argument("--tau", type=float, default=0, help="Threshold for the detector")
    parser.add_argument("--norm", type=str, default='linf', help="Norm for the attack")
    parser.add_argument("--attack_bitstring", action="store_true",
                        help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='',
                        choices=['audioseal', 'wavmark', 'timbre', 'silentcipher', 'patchwork', 'qim',  'echo'], help="Model to be attacked")

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


class logger():
    def __init__(self, filename, raw, idx):
        self.visqol = api_visqol()
        self.raw = raw
        self.best_l2 = 1000
        self.best_linf = 1000
        self.best_acc = 1
        self.best_snr = -1000
        self.best_visqol = -1
        self.idx = idx
        file_exists = os.path.exists(filename)
        self.log = open(filename, 'a' if file_exists else 'w')
        if not file_exists:
            # Write the title only if the file does not exist
            self.log.write(
                'idx, query, l2, linf, acc, snr, visqol, best_l2, best_linf, best_acc, best_snr, best visqol\n')

    def evaluate(self, signal, query, acc):
        if_better = False
        l2 = np.linalg.norm(signal - self.raw)
        linf = np.max(np.abs(signal - self.raw))
        snr = 10 * np.log10(np.sum(np.square(self.raw)) / np.sum(np.square(signal - self.raw)))
        visqol_score = self.visqol.Measure(np.array(self.raw.squeeze(), dtype=np.float64),
                                           np.array(signal.squeeze(), dtype=np.float64)).moslqo
        if l2 < self.best_l2:
            self.best_l2 = l2
        if linf < self.best_linf:
            self.best_linf = linf
        if acc < self.best_acc:
            self.best_acc = acc
        if visqol_score > self.best_visqol:
            self.best_visqol = visqol_score
            if_better = True
        if snr >= self.best_snr:
            self.best_snr = snr
        self.log.write(
            f'{self.idx}, {query}, {l2}, {linf}, {acc}, {snr}, {visqol_score}, {self.best_l2}, {self.best_linf}, {self.best_acc}, {self.best_snr}, {self.best_visqol}\n')
        return if_better





class EchoHidingDecoderModel(nn.Module):
    """Dummy model to satisfy ART requirements"""

    def forward(self, x):
        return x


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
        # Convert to numpy array
        x_np = x.squeeze().cpu().numpy()

        # Convert to mono if needed
        if len(x_np.shape) > 1:
            x_np = x_np[:, 0]

        N = len(x_np) // self.L  # Number of frames
        if N == 0:
            return torch.zeros(self.msg_length, device=x.device)

        # Reshape into frames (column-major order)
        xsig = x_np[:N * self.L].reshape(self.L, N, order='F')
        detected_bits = []

        for k in range(min(N, self.msg_length)):
            # Compute cepstrum
            spectrum = np.abs(np.fft.fft(xsig[:, k]))
            log_spectrum = np.log(spectrum + np.finfo(float).eps)
            rceps = np.fft.ifft(log_spectrum).real

            # Extract peaks at delay locations
            peak0 = abs(rceps[self.d0])
            peak1 = abs(rceps[self.d1])

            # Decision logic
            detected_bits.append(0 if peak0 >= peak1 else 1)

        # Pad with zeros if we didn't get enough bits
        while len(detected_bits) < self.msg_length:
            detected_bits.append(0)

        return torch.tensor(detected_bits[:self.msg_length], dtype=torch.float32, device=x.device)



class SilentCipherWrapper(nn.Module):
    def __init__(self, silentcipher_model):
        super(SilentCipherWrapper, self).__init__()
        self.model = silentcipher_model

    def encode_wav(self, *args, **kwargs):
        return self.model.encode_wav(*args, **kwargs)

    def decode_wav(self, *args, **kwargs):
        return self.model.decode_wav(*args, **kwargs)


class QIMDetectorWrapper(nn.Module):
    def __init__(self, delta=0.01, msg_length=16, step=100):
        super().__init__()
        self.delta = delta
        self.msg_length = msg_length
        self.step = step

    def forward(self, x):
        # Convert to numpy for processing
        x_np = x.squeeze().cpu().numpy()
        detected_bits = []

        # Extract samples at regular intervals
        for i in range(0, len(x_np), self.step):
            if len(detected_bits) >= self.msg_length:
                break

            sample = x_np[i]

            # QIM detection logic
            z0 = np.round(sample / self.delta) * self.delta + (-1) ** (0 + 1) * self.delta / 4.
            z1 = np.round(sample / self.delta) * self.delta + (-1) ** (1 + 1) * self.delta / 4.

            d0 = np.abs(sample - z0)
            d1 = np.abs(sample - z1)

            detected_bits.append(0 if d0 < d1 else 1)

        return torch.tensor(detected_bits[:self.msg_length], dtype=torch.float32, device=x.device)


class QIMDetectorModel(nn.Module):
    """Dummy model to satisfy ART requirements"""

    def forward(self, x):
        return x


class PatchworkModel(nn.Module):
    """Dummy model to satisfy ART requirements"""

    def forward(self, x):
        return x


class PatchworkWrapper(nn.Module):
    def __init__(self, watermark_length=16, sr=16000, fs=3000, fe=7000, k1=0.195, k2=0.08):
        super().__init__()
        self.watermark_length = watermark_length
        self.sr = sr
        self.fs = fs
        self.fe = fe
        self.k1 = k1
        self.k2 = k2

    def forward(self, x):
        # Convert to numpy for processing
        x_np = x.squeeze().cpu().numpy()
        L = len(x_np)

        # Compute DCT
        X = dct(x_np, type=2, norm='ortho')

        # Frequency bounds to indices
        si = int(self.fs / (self.sr / L))
        ei = int(self.fe / (self.sr / L))
        Xs = X[si:(ei + 1)]
        Ls = len(Xs)

        # Adjust length
        if Ls % (self.watermark_length * 2) != 0:
            Ls -= Ls % (self.watermark_length * 2)
            Xs = Xs[:Ls]

        # Create paired segments
        Xsp = np.dstack((Xs[:Ls // 2], Xs[:(Ls // 2 - 1):-1])).flatten()
        segments = np.array_split(Xsp, self.watermark_length * 2)

        # Calculate bits
        watermark_bits = []
        for i in range(0, len(segments), 2):
            j = (i // 2) + 1
            rj = self.k1 * np.exp(-self.k2 * j)

            m1j = np.mean(np.abs(segments[i]))
            m2j = np.mean(np.abs(segments[i + 1]))

            if (m1j - m2j) * rj >= 0:
                watermark_bits.append(0)
            else:
                watermark_bits.append(1)

        return torch.tensor(watermark_bits, dtype=torch.float32, device=x.device)


# Define a wrapper for the watermark detector to fit into the HopSkipJump interface
class WatermarkDetectorWrapper(PyTorchClassifier):
    def __init__(self, model, message, detector_type, on_bitstring, th, model_type, device):
        if model_type == 'patchwork':
            model = PatchworkModel()
        if model_type == 'qim':
            model = QIMDetectorModel()

        if model_type == 'echo':
            model = EchoHidingDecoderModel()


        super(WatermarkDetectorWrapper, self).__init__(model=model,
                                                       input_shape=(1,), nb_classes=2, channels_first=True, loss=None)
        self._device = device
        self.message = message.to(self._device)
        self.detector_type = detector_type
        self.th = th
        self.on_bitstring = on_bitstring
        if model_type == 'patchwork':
            self.patchwork = PatchworkWrapper(watermark_length=len(message)).to(device)
        if model_type == 'qim':
            self.qim = QIMDetectorWrapper(delta=0.01, msg_length=len(message)).to(device)


        if model_type == 'echo':
            self.echo = EchoHidingWrapper(msg_length=len(message)).to(device)


        elif model is not None:  # Only try to move to device if model exists
            self.model.to(self._device)
        if model_type == 'timbre':
            self.predict = self.predict_timbre
            self.bwacc = self.bwacc_timbre
        elif model_type == 'wavmark':
            self.predict = self.predict_wavmark
            self.bwacc = self.bwacc_wavmark
        elif model_type == 'audioseal':
            self.predict = self.predict_audioseal
            self.bwacc = self.bwacc_audioseal
        elif model_type == 'silentcipher':
            self.predict = self.predict_silentcipher
            self.bwacc = self.bwacc_silentcipher

        elif model_type == 'patchwork':
            self.predict = self.predict_patchwork
            self.bwacc = self.bwacc_patchwork

        elif model_type == 'qim':
            self.predict = self.predict_qim
            self.bwacc = self.bwacc_qim

        elif model_type == 'echo':
            self.predict = self.predict_echo
            self.bwacc = self.bwacc_echo



    def predict_echo(self, signal, batch_size=1):
        signal_t = torch.tensor(signal, dtype=torch.float32, device=self._device)
        detected_bits = self.echo(signal_t)

        if self.on_bitstring:
            return self.our_conversion_logic(detected_bits)
        else:
            bit_acc = (detected_bits == self.message).float().mean()
            return self.our_conversion_logic_binary(bit_acc.item())

    def bwacc_echo(self, signal):
        detected_bits = self.echo(signal)
        bit_acc = (detected_bits == self.message).float().mean()
        return bit_acc.item() if not self.on_bitstring else bit_acc.item()



    def predict_qim(self, signal, batch_size=1):
        signal = torch.tensor(signal, dtype=torch.float).to(self._device)
        detected_bits = self.qim(signal)

        if self.on_bitstring:
            return self.our_conversion_logic(detected_bits)
        else:
            # Calculate bit accuracy as confidence
            bit_acc = (detected_bits == self.message).float().mean()
            return self.our_conversion_logic_binary(bit_acc.item())

    def bwacc_qim(self, signal):
        detected_bits = self.qim(signal)

        if self.on_bitstring:
            bit_acc = (detected_bits == self.message).float().mean()
            return bit_acc.item()
        else:
            return (detected_bits == self.message).float().mean().item()

    def predict_patchwork(self, signal, batch_size=1):
        signal = torch.tensor(signal, dtype=torch.float).to(self._device)
        detected_bits = self.patchwork(signal)

        if self.on_bitstring:
            return self.our_conversion_logic(detected_bits)
        else:
            # Calculate bit accuracy as confidence
            bit_acc = (detected_bits == self.message).float().mean()
            return self.our_conversion_logic_binary(bit_acc.item())

    def bwacc_patchwork(self, signal):
        detected_bits = self.patchwork(signal)

        if self.on_bitstring:
            bit_acc = (detected_bits == self.message).float().mean()
            return bit_acc.item()
        else:
            # For confidence mode, use the proportion of matching bits
            return (detected_bits == self.message).float().mean().item()

    def predict_silentcipher(self, signal, batch_size=1):  # signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).squeeze(0).numpy()
        
        result = self.model.decode_wav(signal, 16000, phase_shift_decoding=False)

        if self.on_bitstring:
            if result['status'] != True:
                return np.array([[1, 0]])
            # Convert byte message to bits
            binary_sequence = ''.join(format(byte, '08b') for byte in result['messages'][0][:2])

            # Extract the first 16 bits
            first_16_bits = binary_sequence[:16]
            print(first_16_bits)
            msg_bits = torch.tensor(list(map(int, first_16_bits)), dtype=torch.int)
            
            return self.our_conversion_logic(msg_bits)
        else:
            return self.our_conversion_logic_binary(result['confidences'][0] if result['status'] == True else None)

    def bwacc_silentcipher(self, signal):  # signal is tensor on gpu
        signal = signal.squeeze(0).cpu().numpy()
        result = self.model.decode_wav(signal, 16000, phase_shift_decoding=False)
        print(result)
        if self.on_bitstring:
            if result['status'] != True or result['messages'][0] is None:
                return 0
            # Convert byte message to bits
            binary_sequence = ''.join(format(byte, '08b') for byte in result['messages'][0][:2])

            # Extract the first 16 bits
            first_16_bits = binary_sequence[:16]
            print(first_16_bits)
            msg_bits = torch.tensor(list(map(int, first_16_bits)), dtype=torch.int)
            bit_acc = 1 - torch.sum(torch.abs(msg_bits - self.message)) / self.message.shape[0]
            return bit_acc.item()
        else:
            return result['confidences'][0] if result['status'] == True else 0

    def predict_audioseal(self, signal, batch_size=1):  # signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).to(self._device)
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            return self.our_conversion_logic(msg_decoded)
        else:
            return self.our_conversion_logic_binary(result)

    def bwacc_audioseal(self, signal):  # signal is tensor on gpu
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None:
                return 0
            else:
                msg_decoded = torch.tensor(msg_decoded, dtype=torch.int).to(self.message.device)
                bit_acc = 1 - torch.sum(torch.abs(msg_decoded - self.message)) / self.message.shape[0]
                return bit_acc.item()
        else:
            return result

    def predict_wavmark(self, signal, batch_size=1):  # signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).squeeze(0)
        payload, info = wavmark.decode_watermark(self.model, signal)  # signal: [,80000]
        return self.our_conversion_logic(payload)

    def bwacc_wavmark(self, signal):  # signal is tensor on gpu
        signal = signal.squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)  # signal: [,80000]
        if payload is None:
            return 0
        else:
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1 - torch.sum(torch.abs(payload - self.message)) / self.message.shape[0]
            return bit_acc.item()

    def predict_timbre(self, signal, batch_size=1):  # signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).to(self._device)
        payload = self.model.test_forward(signal.unsqueeze(0))  # signal: [1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        if self.detector_type == 'double-tailed':
            class_idx = torch.logical_or((bit_acc >= self.th), (bit_acc <= (1 - self.th)))
        if self.detector_type == 'single-tailed':
            class_idx = (bit_acc >= self.th)
        return np.array([[0, 1]]) if class_idx else np.array([[1, 0]])

    def bwacc_timbre(self, signal):  # signal is tensor on gpu
        payload = self.model.test_forward(signal.unsqueeze(0))  # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return bit_acc

    def our_conversion_logic(self, payload):
        if payload is None:
            return np.array([[1, 0]])  # NOTE if result is None, then it is classified as 0, not detected
        else:
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1 - torch.sum(torch.abs(payload - self.message)) / self.message.shape[0]
            if self.detector_type == 'double-tailed':
                class_idx = torch.logical_or((bit_acc >= self.th), (bit_acc <= (1 - self.th))).long()
            if self.detector_type == 'single-tailed':
                class_idx = (bit_acc >= self.th).long()
            return np.array([[0, 1]]) if class_idx else np.array([[1, 0]])

    def our_conversion_logic_binary(self, bit_acc):
        if bit_acc is None:
            return np.array([[1, 0]])  # NOTE if result is None, then it is classified as 0, not detected
        else:
            if self.detector_type == 'double-tailed':
                class_idx = torch.logical_or((bit_acc >= self.th), (bit_acc <= (1 - self.th)))
            if self.detector_type == 'single-tailed':
                class_idx = (bit_acc >= self.th)
            return np.array([[0, 1]]) if class_idx else np.array([[1, 0]])


def initial_ad_samples(model, signal, query, tau):
    signal_power = torch.mean(signal ** 2)
    for noise_level in [30, 25, 20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 0]:
        noise_power = signal_power / (10 ** (noise_level / 10))
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal) * noise_std
        adv_signal = signal + noise
        query = query + 1
        acc = model.bwacc(adv_signal)
        if acc <= tau:
            break
    snr = 10 * torch.log10(torch.mean(signal ** 2) / torch.mean(noise ** 2))
    print(f'detection probability: {acc}, snr: {snr:.3f}')
    return adv_signal, query


def decode_audio_files_perturb_blackbox(model, args, device):
    # NOTE: this is the subsampled dataset of audiomarkdataBench
    #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
    #progress_bar = tqdm(file_list, desc=f"black-box-perturbation")
    save_path = os.path.join(args.blackbox_folder, "HSJA_signal")
    os.makedirs(save_path, exist_ok=True)
    watermarked_file = os.path.basename(args.input_dir)
    
    idx = '_'.join(watermarked_file.split('_')[2:])  # idx_bitstring_snr
    print(idx)
    #path = os.path.join(output_dir, watermarked_file)
    waveform, sample_rate = torchaudio.load(args.input_dir)
    '''waveform.shape = [1, 80000]'''
    waveform = waveform.to(device=device)
    original_payload_str = watermarked_file.split('_')[1]

    original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.int)

    print(original_payload_str)

    detector = WatermarkDetectorWrapper(model, original_payload, 'single-tailed', args.attack_bitstring, args.tau,
                                        args.model, device)
    adv_signal, num_queries = initial_ad_samples(detector, waveform, np.zeros((1)), args.tau)
    # Initialize the HopSkipJump attack;
    attack = HopSkipJump(classifier=detector, targeted=False, norm=args.norm, max_iter=args.max_iter,
                            max_eval=args.max_eval, init_eval=args.init_eval, batch_size=args.batch_size)
    waveform = waveform.detach().cpu().numpy()
    adv_signal = adv_signal.detach().cpu().numpy()
    # log = logger(filename=os.path.join(save_path, f'{args.model}_tau{args.tau}.csv'), raw=waveform, idx=idx)
    while num_queries <= args.query_budget and num_queries >= 0:
        adv_signal, num_queries = attack.generate(x=waveform, x_adv_init=adv_signal, num_queries_ls=num_queries,
                                                    resume=True)
        print(f'num_queries: {num_queries}')
        acc = detector.bwacc(torch.tensor(adv_signal, dtype=torch.float).to(device))
        # if log.evaluate(signal=adv_signal, query=num_queries[0], acc=acc):
        #   print(f'idx: {idx}, query: {num_queries[0]}, acc: {acc:.3f}, snr: {log.best_snr:.1f}, visqol: {log.best_visqol:.3f}')
    torchaudio.save(os.path.join(save_path,
                                     watermarked_file),
                    torch.tensor(adv_signal), sample_rate)


def main():
    args = parse_arguments()

    if args.norm == 'l2':
        args.norm = 2
    else:
        args.norm = np.inf

    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
        #output_dir = 'path to the audioseal watermarked'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
        #output_dir = 'path to the wavmark watermarked'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'timbre':
        process_config = yaml.load(open("Timbre_10/config/process.yaml", "r"),
                                   Loader=yaml.FullLoader)
        model_config = yaml.load(open("Timbre_10/config/model.yaml", "r"),
                                 Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        detector = Decoder(process_config, model_config, 10, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder,
                           attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load(
            'Timbre_10/results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar')
        detector.load_state_dict(checkpoint['decoder'], strict=False)
        detector.eval()
        model = detector
        #output_dir = 'path to the Timbre watermarked'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'silentcipher':

        sc_model = silentcipher.get_model(model_type='44.1k', device=device)
        model = SilentCipherWrapper(sc_model)  
        #output_dir = 'path to the silentcipher watermarked'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'patchwork':
       
        model = None  
        #output_dir = 'path to the Patchwork watermarked'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'qim':
        model = None  
        #output_dir = 'path to the QIM watermarked'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'echo':
        model = None  # The wrapper handles the detection
        #output_dir = 'path to the Echo watermarked'
        #os.makedirs(output_dir, exist_ok=True)


    decode_audio_files_perturb_blackbox(model,args, device)


if __name__ == "__main__":
    main()
