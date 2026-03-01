# VoxWatermark

VoxWatermark is a benchmark for audio watermark detection under no-box, black-box, and white-box perturbations.

This repository provides:
- Watermarked dataset generation scripts (`datasets/`)
- Dataset splitting + perturbation generation (`Splitting_Dataset/`)
- Two baseline detectors (`baseline/AudioWMD.py`, `baseline/WMD.py`)

## Repository Structure

```text
VoxWatermark/
├── datasets/                  # watermark embedding scripts + dataset source notes
├── Splitting_Dataset/         # split + perturbation pipeline
├── baseline/                  # baseline detectors
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
cd /path/to/VoxWatermark
pip install -r requirements.txt
```

Optional (for extra perturbation dependencies used by some scripts):

```bash
pip install -r datasets/requirements.txt
```

## Step 1: Prepare Clean Data and Generate Watermarked Data

### 1.1 Clean data sources

Clean (unwatermarked) data links used in this project:
- LibriSpeech: https://drive.google.com/drive/folders/10chT3x4a4nZIN4TSaksyJsPIprC1o-wR?usp=drive_link
- Common Voice: https://drive.google.com/drive/folders/1RdtmzeXqllTIDTECBeaBCVeC659zkN3_?usp=drive_link
- ASVSpoof5: https://drive.google.com/drive/folders/1OWrhNmMa-sLfiCN04PozMxn5CmfVBxOC?usp=drive_link
- VCTK: https://drive.google.com/drive/folders/1XkAEMTii85adRHR965apSgNZvt72HdC-?usp=drive_link
- AISHELL-1: https://drive.google.com/drive/folders/19HfEndmJw14cg7Zp45hiiV1HdxBBA1tI?usp=drive_link

### 1.2 Generate watermarked audio

Most scripts under `datasets/*/*.py` are standalone and use hardcoded `INPUT_DIR` / `OUTPUT_DIR` constants.

You should:
1. Open each script and set `INPUT_DIR` to your clean-audio folder.
2. Set `OUTPUT_DIR` to the target watermark-method folder.
3. Run the script.

Common embedding scripts:
- `datasets/Audioseal/audioseal.py`
- `datasets/Wavmark/wavmark.py`
- `datasets/SilentCipher/SC.py`
- `datasets/Perth/perth.py`
- `datasets/Patchwork/patchwork.py`
- `datasets/Echo-hiding_single_kernel/echo.py`
- `datasets/QuantizationIndexModulation/QIM.py`
- `datasets/Spread-Spectrum/DSSS.py`
- `datasets/Phase_coding/phase.py`
- `datasets/LSB/lsb.py`
- `datasets/Timbre_10/timbre.py`

Example:

```bash
python datasets/Audioseal/audioseal.py
python datasets/Wavmark/wavmark.py
python datasets/Patchwork/patchwork.py
```

## Step 2: Split Dataset + Generate Perturbations

The main entry is:
- `Splitting_Dataset/split_speech.py`

It does all of the following:
- Build train/validation/test splits from clean corpora
- Match each clean file with one watermarked version
- Add no-box perturbations
- Add black-box perturbations
- Add white-box perturbations
- Export `dataset_manifest.csv`

### 2.1 Configure input roots

In `Splitting_Dataset/split_speech.py`, edit:

- `CLEAN_ROOTS` (paths to clean LibriSpeech/CommonVoice/AISHELL/VCTK)
- (optional) counts/ratios/attack lists near the top:
  - `TRAIN_CLEAN_COUNT`, `VAL_RATIO`, `TEST1_COUNT`, `TEST2_COUNT`
  - `TRAIN_WM_METHODS`, `TEST_WM_METHODS`
  - `TEST_NO_BOX_PERTURBS`, `TEST_BLACK_BOX_PERTURBS`

### 2.2 Noise assets for no-box perturbation

Extract the noise archive before running split/perturbation pipeline:

```bash
cd Splitting_Dataset
unzip noises_wav.zip
```

If needed, adjust paths in `Splitting_Dataset/no_box_funcs.py` to your local extracted noise location.

### 2.3 Run split + perturbation pipeline

```bash
cd /path/to/VoxWatermark/Splitting_Dataset
python split_speech.py
```

Output:
- `Splitting_Dataset/dataset_manifest.csv`

### 2.4 Data hierarchy (integrated from `Splitting_Dataset/README.md`)

```text
datasets/
  LibriSpeech/
    clean/
      audio1.wav
      perturbations/
        no_box/
          time_jitter/
            audio1_pert_time_jitter_p1.wav
    LSB/
      wm_audio1.wav
      perturbations/
        no_box/
          time_jitter/
            wm_audio1_pert_time_jitter_p1.wav
    QIM/
      wm_audio2.wav
      perturbations/
        ...
    DSSS/
      wm_audio3.wav
      perturbations/
        ...
```

### 2.5 Manual perturbation scripts (optional)

If you want to run perturbations manually for one file, scripts are in `Splitting_Dataset/`:
- `black-box-HSJA_signal.py`
- `black-box-HSJA_spectrogram.py`
- `black-box_square.py`
- `white-box_removal.py`
- `white-box_forgery.py`

Example command:

```bash
cd /path/to/VoxWatermark/Splitting_Dataset
python black-box-HSJA_signal.py \
  --gpu 0 \
  --input_dir /abs/path/to/watermarked.wav \
  --model wavmark \
  --query_budget 10000 \
  --tau 0.15 \
  --norm linf \
  --blackbox_folder /abs/path/to/output_dir
```

## Step 3: Run Baselines

Baselines are:
- `baseline/AudioWMD.py`
- `baseline/WMD.py`

Use the manifest produced in Step 2.

### 3.1 Baseline A: AudioWMD

```bash
cd /path/to/VoxWatermark
mkdir -p baseline/saved_models_binary_multi
python baseline/AudioWMD.py \
  --manifest Splitting_Dataset/dataset_manifest.csv \
  --train_split train \
  --val_split validation \
  --queries 8 \
  --save_base baseline/saved_models_binary_multi/audiowmd_base.pth \
  --save_meta baseline/saved_models_binary_multi/audiowmd_meta.pkl
```

### 3.2 Baseline B: WMD

```bash
cd /path/to/VoxWatermark
mkdir -p baseline/saved_models_binary_multi
python baseline/WMD.py \
  --manifest Splitting_Dataset/dataset_manifest.csv \
  --det_split train \
  --clean_split train \
  --val_split validation \
  --save_path baseline/saved_models_binary_multi/wmd_strict.pth
```

## Notes

- The dataset-generation scripts are heterogeneous and several rely on fixed constants in code. For reproducibility, keep a record of every `INPUT_DIR`/`OUTPUT_DIR` you set.
- `split_speech.py` assumes watermarked filenames keep payload/method information in the filename.
- GPU is recommended for black-box/white-box perturbations and baseline training.
