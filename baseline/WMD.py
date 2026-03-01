#!/usr/bin/env python3
"""
Strict WMD-style training (paper algorithm) for audio data.

Paper-faithful parts:
- Asymmetric loss: L_total = L_sm(clean) + L_lin(detection)
- Iterative sample pruning on detection set (not weight pruning)
- Reinitialize detector after each pruning step
- AdamW with paper-default core hyperparameters

Only intentional adaptation:
- Data reading follows needles_audio_blackbox_xlsr_conv_dprnn.py CSV schema
  (split, derived_path, is_watermarked)
- Audio input is log-mel spectrogram, detector backbone is ConvNeXt-V2 style
  with 5 ConvNeXt-V2 blocks (paper backbone family).
"""
import os
import csv
import json
import time
import math
import argparse
import random
from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

CFG = {
    "manifest": "WAD/gen/dataset_manifest.csv",
    "det_split": "train",
    "clean_split": "train",
    "val_split": "validation",
    "sample_rate": 16000,
    "duration": 3.0,
    "n_fft": 1024,
    "hop": 320,
    "n_mels": 128,
    "fixed_frames": 160,
    "batch_size": 32,
    "num_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dim": 96,
    "drop_path": 0.0,
    "tau": 1.0,
    "pruning_rate": 0.10,
    "pruning_interval": 10,
    "log_every": 50,
    "log_file": None,
    "save_path": "./saved_models_binary_multi/convnextv2_wmd_strict.pth",
    "cache_dir": None,
    "thr_strategy": "tpr_fpr",
    "fixed_thr": 0.5,
}


def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    if len(y) < tgt:
        y = np.pad(y, (0, tgt - len(y)))
    else:
        y = y[:tgt]
    return y


def wav_to_logmel(y, cfg):
    m = librosa.feature.melspectrogram(
        y=y,
        sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop"],
        n_mels=cfg["n_mels"],
        fmin=0,
        fmax=cfg["sample_rate"] // 2,
    )
    m = librosa.power_to_db(m + 1e-10, ref=np.max)
    m = (m - m.mean()) / (m.std() + 1e-6)
    t = int(cfg["fixed_frames"])
    if m.shape[1] < t:
        m = np.pad(m, ((0, 0), (0, t - m.shape[1])))
    else:
        m = m[:, :t]
    return m.astype(np.float32)


class AudioItems:
    def __init__(self, manifest: str, split: str):
        items = []
        with open(manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                p = row.get("derived_path")
                if not p or not os.path.isfile(p):
                    continue
                y = int(row.get("is_watermarked", 0))
                items.append((p, y))
        self.items = items


class DetectionSubset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], cfg, indices: List[int]):
        self.items = items
        self.cfg = cfg
        self.indices = list(indices)

    def set_indices(self, indices: List[int]):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        p, y = self.items[j]
        mel = wav_to_logmel(load_audio(p, self.cfg), self.cfg)
        return torch.tensor(mel[None, ...], dtype=torch.float32), float(y), j


class CleanDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], cfg):
        self.items = [(p, y) for (p, y) in items if y == 0]
        self.cfg = cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        mel = wav_to_logmel(load_audio(p, self.cfg), self.cfg)
        return torch.tensor(mel[None, ...], dtype=torch.float32), float(y)


def collate_det(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)


def collate_clean(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.float32)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x + residual


class ConvNeXtV2Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["model_dim"])
        self.stem = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=4, stride=4),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(d) for _ in range(5)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        return self.head(x).squeeze(-1)


def init_detector(cfg):
    model = ConvNeXtV2Detector(cfg).to(torch.device(cfg["device"]))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    return model, opt


def score_detection_set(det_items, active_indices, cfg, model, batch_size):
    ds = DetectionSubset(det_items, cfg, active_indices)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_det)
    scores = {}
    model.eval()
    with torch.no_grad():
        for xs, _, ids in dl:
            s = model(xs.to(torch.device(cfg["device"]))).detach().cpu().numpy().tolist()
            for j, v in zip(ids.tolist(), s):
                scores[int(j)] = float(v)
    return scores


def asymmetric_loss(pc: torch.Tensor, pd: torch.Tensor, tau: float):
    # Paper Eq. (8)-(9): L_total = L_sm(pc) + L_lin(pd)
    l_sm = tau * torch.logsumexp(pc / tau, dim=0)
    l_lin = -pd.mean()
    return l_sm + l_lin, l_sm, l_lin


def run_eval(model, cfg, log):
    device = torch.device(cfg["device"])
    det_items = AudioItems(cfg["manifest"], cfg["det_split"]).items
    val_items = AudioItems(cfg["manifest"], cfg["val_split"]).items

    if len(det_items) == 0:
        raise RuntimeError("Detection split has no valid samples.")
    if len(val_items) > 0:
        eval_items = val_items
        eval_name = cfg["val_split"]
    else:
        eval_items = det_items
        eval_name = cfg["det_split"]
        log("Warning: validation split is empty. Final eval will use detection split.")

    eval_ds = DetectionSubset(eval_items, cfg, list(range(len(eval_items))))
    eval_dl = DataLoader(eval_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, collate_fn=collate_det)
    model.eval()
    all_s, all_y = [], []
    with torch.no_grad():
        for xs, ys, _ in eval_dl:
            s = model(xs.to(device)).detach().cpu().numpy().tolist()
            all_s.extend(s)
            all_y.extend(ys.numpy().tolist())

    try:
        auroc = roc_auc_score(all_y, all_s)
    except Exception:
        auroc = 0.5
    log(f"[eval:{eval_name}] score-AUROC: {auroc:.4f}")

    scores = np.asarray(all_s, dtype=np.float32)
    labels = np.asarray(all_y, dtype=np.int64)
    if cfg["thr_strategy"] == "fixed":
        best_thr = float(cfg["fixed_thr"])
    elif cfg["thr_strategy"] == "f1_watermark":
        best_thr = 0.0
        best_f1 = -1.0
        for t in np.linspace(float(scores.min()), float(scores.max()), 201):
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            f1 = 2 * p * r / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)
    elif cfg["thr_strategy"] == "tpr_at_fpr10":
        fpr, tpr, thr = roc_curve(labels, scores)
        valid = np.where(fpr <= 0.10)[0]
        if len(valid) == 0:
            idx = int(np.argmin(np.abs(fpr - 0.10)))
        else:
            idx = int(valid[np.argmax(tpr[valid])])
        best_thr = float(thr[idx])
        log(f"[eval:{eval_name}] TPR@10%FPR: {float(tpr[idx]):.4f}  (FPR={float(fpr[idx]):.4f})")
    elif cfg["thr_strategy"] == "fpr_at_tpr90":
        fpr, tpr, thr = roc_curve(labels, scores)
        valid = np.where(tpr >= 0.90)[0]
        if len(valid) == 0:
            idx = int(np.argmin(np.abs(tpr - 0.90)))
        else:
            idx = int(valid[np.argmin(fpr[valid])])
        best_thr = float(thr[idx])
        log(f"[eval:{eval_name}] FPR@90%TPR: {float(fpr[idx]):.4f}  (TPR={float(tpr[idx]):.4f})")
    else:
        fpr, tpr, thr = roc_curve(labels, scores)
        best_thr = float(thr[int(np.argmax(tpr - fpr))])
    pred = (scores >= best_thr).astype(int)
    log(f"[eval:{eval_name}] best_thr: {best_thr:.4f}")
    log(classification_report(labels, pred, target_names=["Clean", "Watermarked"], zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--det_split", type=str, default=None)
    ap.add_argument("--clean_split", type=str, default=None)
    ap.add_argument("--val_split", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--adam_beta1", type=float, default=None)
    ap.add_argument("--adam_beta2", type=float, default=None)
    ap.add_argument("--n_fft", type=int, default=None)
    ap.add_argument("--hop", type=int, default=None)
    ap.add_argument("--n_mels", type=int, default=None)
    ap.add_argument("--fixed_frames", type=int, default=None)
    ap.add_argument("--model_dim", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_path", type=str, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--pruning_rate", type=float, default=None)
    ap.add_argument("--pruning_interval", type=int, default=None)
    ap.add_argument(
        "--thr_strategy",
        type=str,
        default=None,
        choices=["fixed", "tpr_fpr", "f1_watermark", "tpr_at_fpr10", "fpr_at_tpr90"],
    )
    ap.add_argument("--fixed_thr", type=float, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation.")
    ap.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path for --eval_only.")
    args = ap.parse_args()

    cfg = CFG.copy()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    for k in [
        "manifest", "det_split", "clean_split", "val_split",
        "batch_size", "num_epochs", "lr", "weight_decay", "adam_beta1", "adam_beta2",
        "n_fft", "hop", "n_mels", "fixed_frames", "model_dim",
        "device", "save_path",
        "tau", "pruning_rate", "pruning_interval", "thr_strategy", "fixed_thr", "log_every", "log_file",
    ]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if not cfg.get("log_file"):
        base = os.path.splitext(os.path.basename(cfg["save_path"]))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg["log_file"] = f"./logs/{base}_{ts}.log"

    log_fh = None
    if cfg.get("log_file"):
        os.makedirs(os.path.dirname(cfg["log_file"]) or ".", exist_ok=True)
        log_fh = open(cfg["log_file"], "a", encoding="utf-8")

    def log(msg: str):
        print(msg, flush=True)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    device = torch.device(cfg["device"])
    log(f"Using device: {device}")
    log(f"Log file: {cfg['log_file']}")
    log("Config:")
    log(json.dumps(cfg, ensure_ascii=False, sort_keys=True))

    if args.eval_only:
        ckpt_path = args.ckpt_path or cfg["save_path"]
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model, _ = init_detector(cfg)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        log(f"Loaded checkpoint: {ckpt_path}")
        run_eval(model, cfg, log)
        log("Eval-only finished.")
        if log_fh is not None:
            log_fh.close()
        return

    det_items = AudioItems(cfg["manifest"], cfg["det_split"]).items
    clean_items_all = AudioItems(cfg["manifest"], cfg["clean_split"]).items
    val_items = AudioItems(cfg["manifest"], cfg["val_split"]).items
    clean_items = [(p, y) for (p, y) in clean_items_all if y == 0]

    if len(det_items) == 0:
        raise RuntimeError("Detection split has no valid samples.")
    if len(clean_items) == 0:
        raise RuntimeError("Clean split has no clean samples (is_watermarked==0).")
    if len(val_items) == 0:
        log("Warning: validation split is empty. Final eval will use detection split.")

    log(f"Detection samples: {len(det_items)}")
    log(f"Clean samples: {len(clean_items)}")
    log(f"Validation samples: {len(val_items)}")

    model, opt = init_detector(cfg)

    active_indices = list(range(len(det_items)))
    det_ds = DetectionSubset(det_items, cfg, active_indices)
    clean_ds = CleanDataset(clean_items, cfg)

    prune_steps = 0
    last_rank_scores = None
    t_start = time.time()

    for ep in range(1, int(cfg["num_epochs"]) + 1):
        ep_t0 = time.time()
        model.train()
        det_loader = DataLoader(det_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate_det)
        clean_loader = DataLoader(clean_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, collate_fn=collate_clean)

        det_iter = iter(det_loader)
        clean_iter = iter(clean_loader)
        n_steps = max(len(det_loader), len(clean_loader))
        run_loss = 0.0

        for step in range(1, n_steps + 1):
            try:
                xs_d, _, _ = next(det_iter)
            except StopIteration:
                det_iter = iter(det_loader)
                xs_d, _, _ = next(det_iter)
            try:
                xs_c, _ = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_loader)
                xs_c, _ = next(clean_iter)

            pd = model(xs_d.to(device))
            pc = model(xs_c.to(device))
            loss, lsm, llin = asymmetric_loss(pc, pd, float(cfg["tau"]))

            opt.zero_grad()
            loss.backward()
            opt.step()

            run_loss += float(loss.item())
            if step % max(1, int(cfg["log_every"])) == 0:
                log(
                    f"[train] ep {ep}/{cfg['num_epochs']} step {step}/{n_steps} "
                    f"loss {loss.item():.4f} l_sm {lsm.item():.4f} l_lin {llin.item():.4f}"
                )

        log(f"[train] ep {ep}/{cfg['num_epochs']} avg_loss {run_loss / max(1, n_steps):.4f} active_det {len(active_indices)}")
        log(f"[train] ep {ep}/{cfg['num_epochs']} epoch_time_min {(time.time() - ep_t0)/60:.2f}")

        # Avoid pruning/reset at the final epoch so evaluation is not run on a freshly reinitialized model.
        if ep % int(cfg["pruning_interval"]) == 0 and ep < int(cfg["num_epochs"]):
            prune_steps += 1
            rank_scores = score_detection_set(
                det_items=det_items,
                active_indices=list(range(len(det_items))),
                cfg=cfg,
                model=model,
                batch_size=cfg["batch_size"],
            )
            last_rank_scores = rank_scores
            rank_vals = np.array(list(rank_scores.values()), dtype=np.float32)
            log(
                f"[prune] score_stats mean={rank_vals.mean():.4f} std={rank_vals.std():.4f} "
                f"min={rank_vals.min():.4f} max={rank_vals.max():.4f}"
            )

            keep_ratio = (1.0 - float(cfg["pruning_rate"])) ** prune_steps
            keep_n = max(1, int(math.floor(len(det_items) * keep_ratio)))
            ranked = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
            active_indices = [idx for idx, _ in ranked[:keep_n]]
            det_ds.set_indices(active_indices)
            log(
                f"[prune] ep {ep} step {prune_steps} keep_ratio {keep_ratio:.4f} "
                f"keep_n {keep_n}/{len(det_items)}"
            )

            model, opt = init_detector(cfg)
            log("[prune] detector reinitialized")

    # Final ranking scores for detection set.
    if last_rank_scores is None:
        last_rank_scores = score_detection_set(
            det_items=det_items,
            active_indices=list(range(len(det_items))),
            cfg=cfg,
            model=model,
            batch_size=cfg["batch_size"],
        )

    # Save final detector and selected set.
    os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)
    torch.save(
        {
            "cfg": cfg,
            "model": model.state_dict(),
            "selected_detection_indices": active_indices,
            "selected_detection_paths": [det_items[i][0] for i in active_indices],
        },
        cfg["save_path"],
    )
    log(f"Saved checkpoint: {cfg['save_path']}")

    run_eval(model, cfg, log)

    # Report selected-set purity/recall when labels exist in detection split.
    det_labels = np.array([int(y) for _, y in det_items], dtype=np.int64)
    sel_mask = np.zeros(len(det_items), dtype=np.int64)
    sel_mask[np.array(active_indices, dtype=np.int64)] = 1
    tp = int(np.sum((sel_mask == 1) & (det_labels == 1)))
    fp = int(np.sum((sel_mask == 1) & (det_labels == 0)))
    fn = int(np.sum((sel_mask == 0) & (det_labels == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    log(f"[selected-set] size={int(sel_mask.sum())} precision={precision:.4f} recall={recall:.4f}")

    elapsed_min = (time.time() - t_start) / 60.0
    log(f"Total time: {elapsed_min:.2f} min")
    log("Run finished.")

    if log_fh is not None:
        log_fh.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
