import os, sys, math, time, json, argparse, random, pickle
from typing import List, Tuple
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# ----------------- Config -----------------
CFG = {
    "manifest": "WAD/gen/dataset_manifest.csv",
    "train_split": "train",
    "val_split": "validation",   # can be changed to test1_in/test2_in
    "batch_size": 64,
    "num_epochs": 15,
    "lr": 1e-3,
    "sample_rate": 16000,
    "duration": 3.0,
    "n_fft": 2048,
    "hop": 512,
    "n_mels": 64,
    "fixed_frames": 216,
    "queries": 8,
    "save_base": "./saved_models_binary_multi/cnn_base_needles_validation.pth",
    "save_meta": "./saved_models_binary_multi/meta_needles_lr_cnn_validation.pkl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "needles_audio",
    "wandb_entity": None,
    "wandb_run_name": None,
    "thr_strategy": "tpr_fpr",
    "fixed_thr": 0.5,
    "log_file": None,
    "seed": 42,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------- Data -----------------
def load_audio(path, cfg):
    y, _ = librosa.load(path, sr=cfg["sample_rate"])
    tgt = int(cfg["sample_rate"] * cfg["duration"])
    if len(y) < tgt: y = np.pad(y, (0, tgt - len(y)))
    else: y = y[:tgt]
    return y

def wav_to_mel(y, cfg):
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg["sample_rate"], n_fft=cfg["n_fft"], hop_length=cfg["hop"],
        n_mels=cfg["n_mels"], fmin=0, fmax=cfg["sample_rate"]//2
    )
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    T = cfg["fixed_frames"]
    if m.shape[1] < T: m = np.pad(m, ((0,0),(0, T - m.shape[1])))
    else: m = m[:, :T]
    return m.astype(np.float32)

class WADDataset(Dataset):
    def __init__(self, manifest, split, cfg):
        import csv
        self.items = []
        with open(manifest, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split: continue
                p = row.get("derived_path")
                if not p or not os.path.isfile(p): continue
                y = int(row.get("is_watermarked", 0))  # 1=watermarked
                self.items.append((p, y))
        self.cfg = cfg
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        mel = wav_to_mel(load_audio(p, self.cfg), self.cfg)
        return torch.tensor(mel[None, ...]), torch.tensor(y, dtype=torch.long)

# ----------------- Model -----------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Linear(base*4, 1)  # logit for watermarked=1
    def forward(self, x):
        f = self.net(x).flatten(1)
        logit = self.head(f).squeeze(1)
        return logit

# ----------------- Perturbations -----------------
def perturb_wave(y, cfg):
    sr = cfg["sample_rate"]; dur = cfg["duration"]
    p = random.random()
    if p < 0.2:
        shift = random.randint(-int(0.05*len(y)), int(0.05*len(y)))
        y = np.roll(y, shift)
    elif p < 0.4:
        rate = random.uniform(0.92, 1.08)
        y = librosa.effects.time_stretch(y, rate=rate)
    elif p < 0.6:
        steps = random.uniform(-0.4, 0.4)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif p < 0.75:
        y = y + np.random.randn(len(y)) * 0.005
    elif p < 0.9:
        y = y * random.uniform(0.9, 1.1)
    else:
        start = random.randint(0, len(y)-1)
        width = int(0.01 * len(y))
        y[start:start+width] = 0
    tgt = int(sr * dur)
    if len(y) < tgt: y = np.pad(y, (0, tgt - len(y)))
    else: y = y[:tgt]
    return y

# ----------------- Train base CNN -----------------
def train_base(cfg, wb=None):
    device = torch.device(cfg["device"])
    train_set = WADDataset(cfg["manifest"], cfg["train_split"], cfg)
    val_set = WADDataset(cfg["manifest"], cfg["val_split"], cfg)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss()
    best_auc = 0
    for ep in range(cfg["num_epochs"]):
        model.train()
        tot_loss = 0.0
        n_sample = 0
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            logit = model(x)
            loss = bce(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * x.size(0)
            n_sample += x.size(0)
        train_loss = tot_loss / max(1, n_sample)
        # quick val AUROC
        model.eval()
        all_y, all_s = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logit = model(x)
                all_s.extend(torch.sigmoid(logit).cpu().tolist())
                all_y.extend(y.numpy().tolist())
        try:
            auc = roc_auc_score(all_y, all_s)
        except Exception:
            auc = 0.5
        print(f"Epoch {ep+1}/{cfg['num_epochs']} Val AUROC {auc:.4f}")
        if wb is not None:
            wb.log({"epoch": ep + 1, "train_loss": train_loss, "val_auc": auc})
        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "cfg": cfg}, cfg["save_base"])
            print(f"Saved base model to {cfg['save_base']}")
            if wb is not None:
                wb.log({"best_val_auc": best_auc, "epoch": ep + 1})
    return cfg["save_base"]

# ----------------- Query + Meta head -----------------
def query_blackbox(model, device, paths: List[str], cfg, K=8) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    import csv
    meta = {}
    with open(cfg["manifest"], "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta[row["derived_path"]] = int(row.get("is_watermarked", 0))
    total = len(paths); log_every = max(500, total//20) if total else 1
    for i, path in enumerate(paths):
        y0 = load_audio(path, cfg)
        scores = []
        for k in range(K):
            yy = y0 if k == 0 else perturb_wave(y0, cfg)
            mel = wav_to_mel(yy, cfg)
            x = torch.tensor(mel[None, None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                logit = model(x)
                s = torch.sigmoid(logit).item()  # prob watermarked
            scores.append(s)
        scores = np.array(scores)
        f_vec = [
            scores.mean(), scores.std(), scores.max()-scores.min(),
            float((scores > 0.5).mean()),  # ratio predicted as watermarked
            float((np.sign(scores - scores.mean()) != np.sign(scores[0]-0.5)).mean()),  # prediction flip rate
        ]
        feats.append(f_vec)
        labels.append(meta.get(path, 0))
        if (i+1) % log_every == 0 or (i+1) == total:
            print(f"[query] {i+1}/{total} processed", flush=True)
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip_base", action="store_true", help="skip base-model training and load save_base directly")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--train_split", type=str, default=None)
    ap.add_argument("--val_split", type=str, default=None)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--save_base", type=str, default=None)
    ap.add_argument("--save_meta", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument(
        "--thr_strategy",
        type=str,
        default=None,
        choices=["tpr_fpr", "fixed", "tpr_at_fpr10", "fpr_at_tpr90"],
    )
    ap.add_argument("--fixed_thr", type=float, default=None)
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--no_save_meta", action="store_true", help="do not save meta pkl")
    ap.add_argument("--load_meta", type=str, default=None, help="load existing meta pkl and skip meta training")
    args = ap.parse_args()
    cfg = CFG.copy()
    if args.config:
        with open(args.config, "r") as f:
            cfg.update(json.load(f))
    # CLI overrides
    for k in ["batch_size", "num_epochs", "lr", "train_split", "val_split",
              "queries", "manifest", "save_base", "save_meta", "device",
              "wandb_project", "wandb_entity", "wandb_run_name",
              "thr_strategy", "fixed_thr", "log_file", "seed"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    set_seed(int(cfg["seed"]))
    print(f"Seed: {int(cfg['seed'])}")

    tee_fh = None
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    if cfg.get("log_file"):
        os.makedirs(os.path.dirname(cfg["log_file"]) or ".", exist_ok=True)
        tee_fh = open(cfg["log_file"], "a", encoding="utf-8")

        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = Tee(sys.stdout, tee_fh)
        sys.stderr = Tee(sys.stderr, tee_fh)

    wb = None
    if args.use_wandb:
        try:
            import wandb
            wb = wandb.init(project=cfg["wandb_project"],
                            entity=cfg.get("wandb_entity") or None,
                            name=cfg.get("wandb_run_name") or None,
                            config=cfg)
        except ImportError:
            print("wandb not installed; continuing without logging.")
            wb = None

    if not args.skip_base or not os.path.isfile(cfg["save_base"]):
        train_base(cfg, wb=wb)

    # load base model for querying
    device = torch.device(cfg["device"])
    model = SmallCNN().to(device)
    state = torch.load(cfg["save_base"], map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # collect paths
    import csv
    train_paths, val_paths = [], []
    with open(cfg["manifest"], "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("derived_path")
            if not p or not os.path.isfile(p): continue
            if row.get("split") == cfg["train_split"]:
                train_paths.append(p)
            elif row.get("split") == cfg["val_split"]:
                val_paths.append(p)
    if args.load_meta:
        print(f"Train samples: {len(train_paths)}  Val samples: {len(val_paths)}")
        print(f"Loading meta from: {args.load_meta}")
        with open(args.load_meta, "rb") as f:
            meta_obj = pickle.load(f)
        if not isinstance(meta_obj, dict) or "clf" not in meta_obj:
            raise ValueError(f"Invalid meta file: {args.load_meta}")
        clf = meta_obj["clf"]
        meta_cfg = meta_obj.get("cfg", {})
        if isinstance(meta_cfg, dict) and "queries" in meta_cfg:
            print(f"Meta trained with queries={meta_cfg['queries']} | current queries={cfg['queries']}")

        t0 = time.time()
        X_val, y_val = query_blackbox(model, device, val_paths, cfg, cfg["queries"])
        print(f"Query time: {(time.time()-t0)/60:.2f} min")
    else:
        print(f"Train samples: {len(train_paths)}  Val samples: {len(val_paths)}")
        t0 = time.time()
        X_train, y_train = query_blackbox(model, device, train_paths, cfg, cfg["queries"])
        X_val, y_val = query_blackbox(model, device, val_paths, cfg, cfg["queries"])
        print(f"Query time: {(time.time()-t0)/60:.2f} min")

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        if args.no_save_meta:
            print("Skip saving meta pkl (--no_save_meta).")
        else:
            with open(cfg["save_meta"], "wb") as f:
                pickle.dump({"cfg": cfg, "clf": clf}, f)
            print(f"Meta saved to {cfg['save_meta']}")

    val_score = clf.predict_proba(X_val)[:,1]
    auroc = roc_auc_score(y_val, val_score)
    strategy = cfg.get("thr_strategy", "tpr_fpr")
    if strategy == "fixed":
        best_thr = float(cfg["fixed_thr"])
        print(f"Val AUROC: {auroc:.4f}  Fixed thr: {best_thr:.4f}")
    elif strategy == "tpr_at_fpr10":
        fpr, tpr, thr = roc_curve(y_val, val_score)
        valid = np.where(fpr <= 0.10)[0]
        if len(valid) == 0:
            idx = int(np.argmin(np.abs(fpr - 0.10)))
        else:
            idx = int(valid[np.argmax(tpr[valid])])
        best_thr = float(thr[idx])
        print(f"Val AUROC: {auroc:.4f}  TPR@10%FPR: {float(tpr[idx]):.4f}  (FPR={float(fpr[idx]):.4f})  Thr: {best_thr:.4f}")
    elif strategy == "fpr_at_tpr90":
        fpr, tpr, thr = roc_curve(y_val, val_score)
        valid = np.where(tpr >= 0.90)[0]
        if len(valid) == 0:
            idx = int(np.argmin(np.abs(tpr - 0.90)))
        else:
            idx = int(valid[np.argmin(fpr[valid])])
        best_thr = float(thr[idx])
        print(f"Val AUROC: {auroc:.4f}  FPR@90%TPR: {float(fpr[idx]):.4f}  (TPR={float(tpr[idx]):.4f})  Thr: {best_thr:.4f}")
    else:
        fpr, tpr, thr = roc_curve(y_val, val_score)
        best_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thr[best_idx])
        print(f"Val AUROC: {auroc:.4f}  Best thr: {best_thr:.4f}")
    pred = (val_score >= best_thr).astype(int)
    print(classification_report(y_val, pred, target_names=["Clean","Watermarked"], zero_division=0))
    if wb is not None:
        wb.log({"meta_val_auroc": auroc, "meta_best_thr": best_thr})
        wb.finish()
    if tee_fh is not None:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_fh.close()

if __name__ == "__main__":
    main()
