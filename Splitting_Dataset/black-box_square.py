import os
import argparse
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import wavmark
import silentcipher
import torch.nn as nn
import yaml
from Timbre_10.model.conv2_mel_modules import Decoder

from tqdm import tqdm
import fnmatch
import torchaudio.transforms as T
import time
from art.estimators.classification import PyTorchClassifier
from scipy.fftpack import dct, idct


def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    parser.add_argument("--length", type=int, default=5 * 16000, help="Length of the audio samples")

    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input audio files to be attacked")

    
    parser.add_argument("--query_budget", type=int, default=10000, help="Query budget for the attack")
    parser.add_argument("--blackbox_folder", type=str, default="blackbox_square",
                        help="Folder to save the blackbox attack results")

    parser.add_argument("--eps", type=float, default=0, help="Epsilon for the attack")
    parser.add_argument("--p", type=float, default=0.05, help="probability")

    parser.add_argument("--tau", type=float, default=0, help="Threshold for the detector")
    parser.add_argument("--snr", type=list, default=[0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30],
                        help="Signal-to-noise ratio for the attack")
    parser.add_argument("--norm", type=str, default='linf', help="Norm for the attack")
    parser.add_argument("--attack_bitstring", action="store_true",
                        help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--attack_type", type=str, default='both', choices=['amplitude', 'phase', 'both'],
                        help="Type of attack")
    parser.add_argument("--model", type=str, default='',
                        choices=['audioseal', 'wavmark', 'timbre', 'silentcipher', 'patchwork', 'qim',  'echo',
                                 'phase'],
                        help="Model to be attacked")

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


np.set_printoptions(precision=5, suppress=True)


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
        max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def square_attack_linf(model, x, eps, n_iters, p_init, args):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    # min_val, max_val = 0, 1 if x.max() <= 1 else 255
    min_val, max_val = x.min(), x.max()

    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    # x, y = x[corr_classified], y[corr_classified]

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    # logits = model.predict(x_best)
    # loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    # margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    loss = model.get_detection_result(x_best)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    progress_bar = tqdm(range(n_iters), desc='Linf square attack')
    for i_iter in progress_bar:
        idx_to_fool = (loss >= 0)
        # x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        # loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss[idx_to_fool]

        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h + s, center_w:center_w + s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h + s, center_w:center_w + s],
                                        min_val, max_val) - x_best_curr_window) < 10 ** -7) == c * s * s:
                deltas[i_img, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-eps, eps],
                                                                                                  size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        # logits = model.predict(x_new)
        # loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        # margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')
        loss = model.get_detection_result(x_new)

        idx_improved = loss < loss_min_curr
        # loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        # margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        # acc = (margin_min > 0.0).sum() / n_ex_total
        # acc_corr = (margin_min > 0.0).mean()
        # mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
        # avg_margin_min = np.mean(margin_min)
        best_loss = np.minimum(loss, loss_min_curr)
        acc = best_loss.mean()
        time_total = time.time() - time_start
        curr_norms_image = np.max(np.abs(x_new - x))
        curr_norms_image_best = np.max(np.abs(x_best - x))
        progress_bar.set_description(
            f'iter: {i_iter + 1}, acc: {acc:.2f}, time: {time_total:.1f}, max_pert: {curr_norms_image:.2f}, max_pert_best: {curr_norms_image_best:.2f}')
        # print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
        #     format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

        if acc <= args.tau:
            break

    return n_queries, x_best


def square_attack_l2(model, x, eps, n_iters, p_init, args):
    """ The L2 square attack """
    np.random.seed(0)

    min_val, max_val = x.min(), x.max()
    c, h, w = x.shape[1:]

    n_features = c * h * w

    ### initialization
    delta_init = np.zeros(x.shape)
    s = h // 5
    print('Initial square side={} for bumps'.format(s))
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s

    x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, min_val,
                     max_val)

    loss = model.get_detection_result(x_best)

    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    progress_bar = tqdm(range(n_iters))
    for i_iter in progress_bar:
        idx_to_fool = (loss >= 0.0)

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        loss_min_curr = loss[idx_to_fool]
        delta_curr = x_best_curr - x_curr
        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        # norms_window_2 = np.sqrt(
        #     np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
        #            keepdims=True))

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        hps_str = 's={}->{}'.format(s_init, s)
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)

        loss = model.get_detection_result(x_new)

        idx_improved = loss < loss_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        best_loss = np.minimum(loss, loss_min_curr)
        acc = best_loss.mean()
        time_total = time.time() - time_start

        curr_norms_image = np.sqrt(np.sum((x_new - x) ** 2))
        curr_norms_image_best = np.sqrt(np.sum((x_best - x) ** 2))
        progress_bar.set_description(
            f'iter: {i_iter + 1}, acc: {acc:.2f}, time: {time_total:.1f}, max_pert: {curr_norms_image:.2f}, max_pert_best: {curr_norms_image_best:.2f}')

        if acc <= args.tau:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

    return n_queries, x_best


class SilentCipherWrapper(nn.Module):
    def __init__(self, silentcipher_model):
        super(SilentCipherWrapper, self).__init__()
        self.model = silentcipher_model

    def encode_wav(self, *args, **kwargs):
        return self.model.encode_wav(*args, **kwargs)

    def decode_wav(self, *args, **kwargs):
        return self.model.decode_wav(*args, **kwargs)

    def detect_watermark(self, signal):
        # Convert tensor to numpy array if needed
        if isinstance(signal, torch.Tensor):
            signal = signal.squeeze().cpu().numpy()
        result = self.model.decode_wav(signal, 16000, phase_shift_decoding=False)
        return result['confidences'][0] if result['status'] == True else None, result['messages'][0] if result[
            'messages'] else None


class PhaseCodingDecoderModel(nn.Module):
    """Dummy model to satisfy ART requirements"""

    def forward(self, x):
        return x


class PhaseCodingWrapper(nn.Module):
    def __init__(self, msg_length=16, L=1024):
        super().__init__()
        self.msg_length = msg_length
        self.L = L  # Frame length for FFT

    def forward(self, x):
        # Convert to numpy array
        x_np = x.squeeze().cpu().numpy()

        # Convert to mono if needed
        if len(x_np.shape) > 1:
            x_np = x_np[:, 0]

        # Get first frame (pad with zeros if signal is too short)
        if len(x_np) < self.L:
            x_frame = np.pad(x_np, (0, self.L - len(x_np)))
        else:
            x_frame = x_np[:self.L]

        # Compute FFT phase angles
        Phi = np.angle(np.fft.fft(x_frame))

        # Calculate middle point
        mid = self.L // 2

        # Extract bits from phase differences
        detected_bits = []
        for k in range(min(self.msg_length, mid)):
            phase_val = Phi[mid - self.msg_length + k]
            detected_bits.append(0 if phase_val > 0 else 1)

        # Pad with zeros if we didn't get enough bits
        while len(detected_bits) < self.msg_length:
            detected_bits.append(0)

        return torch.tensor(detected_bits[:self.msg_length],
                            dtype=torch.float32,
                            device=x.device)




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




class DsssDecoderModel(nn.Module):
    """Dummy model to satisfy ART requirements"""

    def forward(self, x):
        return x


class DSSSWrapper(nn.Module):
    def __init__(self, msg_length=16, L_min=4096):
        super().__init__()
        self.msg_length = msg_length
        self.L_min = L_min  # Minimum segment length

    def forward(self, x):
        # Convert to numpy array
        x_np = x.squeeze().cpu().numpy()

        # Convert to mono if needed
        if len(x_np.shape) > 1:
            x_np = x_np[:, 0]

        s_len = len(x_np)
        L2 = s_len // self.msg_length
        L = max(self.L_min, L2)  # Segment length
        N = min(s_len // L, self.msg_length)  # Number of segments

        if N == 0:
            return torch.zeros(self.msg_length, device=x.device)

        # Reshape signal into segments (column-major order)
        xsig = x_np[:N * L].reshape(L, N, order='F')

        # Generate reference signal (must match embedding)
        r = np.ones(L)

        # Correlation detection
        detected_bits = []
        for k in range(N):
            correlation = np.sum(xsig[:, k] * r) / L
            detected_bits.append(0 if correlation < 0 else 1)

        # Pad with zeros if needed
        while len(detected_bits) < self.msg_length:
            detected_bits.append(0)

        return torch.tensor(detected_bits[:self.msg_length],
                            dtype=torch.float32,
                            device=x.device)

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


class WatermarkDetectorWrapper(PyTorchClassifier):
    def __init__(self, model, message, detector_type, on_bitstring, transform, th, input_size, model_type, device):
        if model_type == 'patchwork':
            model = PatchworkModel()
        if model_type == 'qim':
            model = QIMDetectorModel()

        if model_type == 'echo':
            model = EchoHidingDecoderModel()
        if model_type == 'dsss':
            model = DsssDecoderModel()
        if model_type == 'phase':
            model = PhaseCodingDecoderModel()
        if model_type == 'dsss':
            model = DsssDecoderModel()


        super(WatermarkDetectorWrapper, self).__init__(model=model,
                                                       input_shape=input_size, nb_classes=2, channels_first=True,
                                                       loss=None)
        self._device = device
        self.message = message.to(self._device)
        self.detector_type = detector_type
        self.th = th
        self.on_bitstring = on_bitstring
        if model_type == 'patchwork':
            self.patchwork = PatchworkWrapper(watermark_length=len(message)).to(device)
        if model_type == 'qim':
            self.qim = QIMDetectorWrapper(delta=0.01, msg_length=len(message)).to(device)
        if model_type == 'phase':
            self.phase = PhaseCodingWrapper(msg_length=len(message)).to(device)
        if model_type == 'echo':
            self.echo = EchoHidingWrapper(msg_length=len(message)).to(device)
        if model_type == 'dsss':
            self.dsss = DSSSWrapper(msg_length=len(message)).to(device)


        self.transform = transform
        self.model.to(self._device)
        if model_type == 'timbre':
            self.get_detection_result = self.get_detection_result_timbre
            self.bwacc = self.bwacc_timbre
        elif model_type == 'wavmark':
            self.get_detection_result = self.get_detection_result_wavmark
            self.bwacc = self.bwacc_wavmark
        elif model_type == 'audioseal':
            self.get_detection_result = self.get_detection_result_audioseal
            self.bwacc = self.bwacc_audioseal
        elif model_type == 'silentcipher':
            self.get_detection_result = self.get_detection_result_silentcipher
            self.bwacc = self.bwacc_silentcipher
        elif model_type == 'patchwork':
            self.get_detection_result = self.get_detection_result_patchwork
            self.bwacc = self.bwacc_patchwork
        elif model_type == 'qim':
            self.get_detection_result = self.get_detection_result_qim
            self.bwacc = self.bwacc_qim
        elif model_type == 'phase':
            self.get_detection_result = self.get_detection_result_phase_coding
            self.bwacc = self.bwacc_phase_coding

        elif model_type == 'echo':
            self.get_detection_result = self.get_detection_result_echo_hiding
            self.bwacc = self.bwacc_echo_hiding
        elif model_type == 'dsss':
            self.get_detection_result = self.get_detection_result_dsss
            self.bwacc = self.bwacc_dsss
            
    def get_detection_result_dsss(self, spectrogram):
        """Wrapper for Echo Hiding detection from spectrogram input."""

        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze()
        detected_msg = self.dsss(signal)

        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return np.array([bit_acc.item()])
        else:
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return np.array([confidence.item()])

    def bwacc_dsss(self, signal):
        """Bit-wise accuracy for Echo Hiding (direct signal input)."""
        detected_msg = self.dsss(signal)
        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return bit_acc
        else:
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return confidence
            

    def get_detection_result_echo_hiding(self, spectrogram):
        """Wrapper for Echo Hiding detection from spectrogram input."""

        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze()
        detected_msg = self.echo(signal)

        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return np.array([bit_acc.item()])
        else:
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return np.array([confidence.item()])

    def bwacc_echo_hiding(self, signal):
        """Bit-wise accuracy for Echo Hiding (direct signal input)."""
        detected_msg = self.echo(signal)
        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return bit_acc
        else:
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return confidence



    def get_detection_result_phase_coding(self, spectrogram):
        """Wrapper for Phase Coding detection from spectrogram input."""
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze()
        detected_msg = self.phase(signal)

        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return np.array([bit_acc.item()])
        else:
            # For Phase Coding, confidence is the fraction of correct bits
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return np.array([confidence.item()])

    def bwacc_phase_coding(self, signal):
        """Bit-wise accuracy for Phase Coding (direct signal input)."""
        detected_msg = self.phase(signal)
        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return bit_acc
        else:
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return confidence

    def get_detection_result_qim(self, spectrogram):
        """Wrapper for QIM detection from spectrogram input."""
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze()
        detected_msg = self.qim(signal)

        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return np.array([bit_acc.item()])
        else:
            # For QIM, confidence can be the fraction of correct bits
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return np.array([confidence.item()])

    def bwacc_qim(self, signal):
        """Bit-wise accuracy for QIM (direct signal input)."""
        detected_msg = self.qim(signal)
        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_msg - self.message)) / len(self.message)
            return bit_acc
        else:
            confidence = 1 - torch.mean(torch.abs(detected_msg - self.message))
            return confidence
    def get_detection_result_patchwork(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze()

        # Convert to numpy for DCT (if needed)
        """if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = signal"""

        detected_bits = self.patchwork(signal)

        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_bits - self.message)) / self.message.shape[0]
            return np.array([bit_acc.item()])
        else:
            # For Patchwork, we can return the average confidence (absolute difference)
            confidence = 1 - torch.mean(torch.abs(detected_bits - self.message))
            return np.array([confidence.item()])

    def bwacc_patchwork(self, signal):
        signal = signal.squeeze()
        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = signal

        detected_bits = self.patchwork(signal_np)

        if self.on_bitstring:
            bit_acc = 1 - torch.sum(torch.abs(detected_bits - self.message)) / self.message.shape[0]
            return bit_acc
        else:
            confidence = 1 - torch.mean(torch.abs(detected_bits - self.message))
            return confidence

    def get_detection_result_silentcipher(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram)
        signal_np = signal.squeeze().cpu().numpy()
        result = self.model.decode_wav(signal_np, 16000, phase_shift_decoding=False)

        if self.on_bitstring:
            if not result.get('status') or not result.get('messages'):
                return np.array([0.0])  # Return float array
            # Convert first 2 bytes to bits (16 bits)
            binary_sequence = ''.join(format(byte, '08b') for byte in result['messages'][0][:2])

            # Extract the first 16 bits
            first_16_bits = binary_sequence[:16]
            
            msg_bits = torch.tensor(list(map(int, first_16_bits)), dtype=torch.int)
            bit_acc = 1 - torch.sum(torch.abs(msg_bits - self.message)) / self.message.shape[0]
            return np.array([bit_acc.item()])  # Return as numpy array
        else:
            confidence = result.get('confidences', [0])[0] if result.get('status') else 0
            return np.array([confidence])


    def bwacc_silentcipher(self, signal):
        signal_np = signal.squeeze().cpu().numpy()
        result = self.model.decode_wav(signal_np, 16000, phase_shift_decoding=False)

        if self.on_bitstring:
            if not result.get('status') or not result.get('messages'):
                return torch.tensor(0.0, device=self._device)  # Return tensor instead of raw int
            # Convert first 2 bytes to bits (16 bits)
            binary_sequence = ''.join(format(byte, '08b') for byte in result['messages'][0][:2])

            # Extract the first 16 bits
            first_16_bits = binary_sequence[:16]
            
            msg_bits = torch.tensor(list(map(int, first_16_bits)), dtype=torch.int)
            bit_acc = 1 - torch.sum(torch.abs(msg_bits - self.message)) / self.message.shape[0]
            return bit_acc  # Already a tensor
        else:
            confidence = result.get('confidences', [0])[0] if result.get('status') else 0
            return torch.tensor(confidence, device=self._device)

    def get_detection_result_audioseal(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram)
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None:
                return np.array([0])
            else:
                bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
                return np.array([bitacc.item()])
        else:
            return np.array([result])

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None:
                return np.array([0])
            else:
                bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
                return np.array([bitacc.item()])
        else:
            return np.array([result])

    def get_detection_result_wavmark(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)
        if payload is None:
            return np.array([0])
        else:
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1 - torch.sum(torch.abs(payload - self.message)) / self.message.shape[0]
            return np.array([bit_acc.item()])

    def bwacc_wavmark(self, signal):
        signal = signal.squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)
        if payload is None:
            return np.array([0])
        else:
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1 - torch.sum(torch.abs(payload - self.message)) / self.message.shape[0]
            return np.array([bit_acc.item()])

    def get_detection_result_timbre(self, spectrogram, batch_size=1):  # signal is np.array
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram)
        payload = self.model.test_forward(signal.unsqueeze(0))  # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bitacc.item()])

    def bwacc_timbre(self, signal):  # signal is tensor on gpu
        payload = self.model.test_forward(signal.unsqueeze(0))  # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bit_acc.item()])


class signal22spectrogram:
    def __init__(self, signal, low_frequency, high_frequency, device, attack_type):
        self.signal = signal
        self.attack_type = attack_type
        self.sig2spec = T.Spectrogram(n_fft=400, power=None).to(device)
        self.spec2sig = T.InverseSpectrogram(n_fft=400).to(device)
        self.spectrogram = self.sig2spec(signal)
        self.amplitude = torch.abs(self.spectrogram)
        self.phase = torch.angle(self.spectrogram)
        self.lf = low_frequency
        self.hf = high_frequency
        self.attack_shape = self.spectrogram[..., low_frequency:high_frequency, :].shape
        self.length = signal.shape[-1]
        '''attack boundary'''

    def signal2spectrogram(self, signal):  # [1,length]->[2 or 1,freq,length]
        spectro_complex = self.sig2spec(signal)
        spectro_amplitude = torch.abs(spectro_complex)
        spectro_phase = torch.angle(spectro_complex)
        if self.attack_type == 'amplitude':
            return spectro_amplitude[..., self.lf:self.hf, :]
        elif self.attack_type == 'phase':
            return spectro_phase[..., self.lf:self.hf, :]
        elif self.attack_type == 'both':
            return torch.cat([spectro_amplitude, spectro_phase], dim=0)[..., self.lf:self.hf, :]

    def spectrogram2signal(self, spectrogram):  # [2,freq,length]->[1,length]
        spectrogram = spectrogram.squeeze()
        if self.attack_type == 'both':
            padding_amp = self.amplitude
            padding_amp[..., self.lf:self.hf, :] = spectrogram[0]
            padding_phase = self.phase
            padding_phase[..., self.lf:self.hf, :] = spectrogram[1]
            spectro_complex = padding_amp * torch.exp(1j * padding_phase)
        elif self.attack_type == 'amplitude':
            padding_amp = self.amplitude
            padding_amp[..., self.lf:self.hf, :] = spectrogram
            spectro_complex = padding_amp * torch.exp(1j * self.phase)
        elif self.attack_type == 'phase':
            padding_phase = self.phase
            padding_phase[..., self.lf:self.hf, :] = spectrogram
            spectro_complex = self.amplitude * torch.exp(1j * padding_phase)
        signal = self.spec2sig(spectro_complex, self.length)
        return signal


def decode_audio_files_perturb_blackbox(model, args, device):
    attack = square_attack_l2 if args.norm == 2 else square_attack_linf
    #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
    #progress_bar = tqdm(file_list, desc=f"black-box-perturbation")
    save_path = os.path.join(args.blackbox_folder, 'square')
    os.makedirs(save_path, exist_ok=True)
    watermarked_file = os.path.basename(args.input_dir)
    # visqol = api_visqol()
    
    parts = watermarked_file.split('_')
    idx = '_'.join(watermarked_file.split('_')[2:])  # idx_bitstring_snr
    #path = os.path.join(output_dir, watermarked_file)
    waveform, sample_rate = torchaudio.load(args.input_dir)

    '''waveform.shape = [1, 80000]'''
    waveform = waveform.to(device=device)
    original_payload_str = watermarked_file.split('_')[1]

    original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.int)

    transform = signal22spectrogram(waveform, 0, 201, device, args.attack_type)
    detector = WatermarkDetectorWrapper(model, original_payload, 'single-tailed', args.attack_bitstring,
                                        transform, args.tau, model_type=args.model,
                                        input_size=transform.attack_shape, device=device)

    '''save to log file'''
    filename = os.path.join(save_path, f'square_spectrogram.csv')
    #log = open(filename, 'a' if os.path.exists(filename) else 'w')
    #log.write('idx, query, acc, snr, visqol\n')

    '''apply the attack to the watermarked signal and save the perturbed signal to the save_path'''
    watermarked_spectrogram = transform.signal2spectrogram(waveform).unsqueeze(0).detach().cpu().numpy()
    n_queries, adv_spectrogram = attack(detector, watermarked_spectrogram, args.eps, args.query_budget, args.p,
                                        args)
    adv_signal = transform.spectrogram2signal(torch.tensor(adv_spectrogram).to(device))
    acc = detector.bwacc(adv_signal).item()
    snr = 10 * torch.log10(torch.sum(waveform ** 2) / torch.sum((adv_signal - waveform) ** 2))
    # visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
    print(f'idx: {idx}, query: {n_queries.item()}, acc: {acc:.3f}, snr: {snr:.1f}')
    #log.write(f'{idx}, {n_queries.item()}, {acc}, {snr}\n')
    torchaudio.save(os.path.join(save_path, watermarked_file),
                    adv_signal.detach().cpu(), sample_rate)


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
        process_config = yaml.load(open("Timbre_10/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("Timbre_10/config/model.yaml", "r"), Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        detector = Decoder(process_config, model_config, 10, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder,
                           attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load('Timbre_10/results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar')
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
        model = None  
        #output_dir = 'path to the Echo watermarked'
        #os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'dsss':
        model = None  
        #output_dir = 'path to the DSSS watermarked'
        #os.makedirs(output_dir, exist_ok=True)

    elif args.model == 'phase':
        model = None  
        #output_dir = 'path to the Phase watermrked'
        #os.makedirs(output_dir, exist_ok=True)

    decode_audio_files_perturb_blackbox(model, args, device)


if __name__ == "__main__":
    main()
