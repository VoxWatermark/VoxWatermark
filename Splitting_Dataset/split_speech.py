import os
import csv
import random
from glob import glob
from collections import Counter, defaultdict
import subprocess
from datasets.perturbations.no_box_funcs import apply_no_box_pert

# Configuration constants
TRAIN_CLEAN_COUNT = 34000
VAL_RATIO = 0.10
TEST1_COUNT = 16000
TEST2_COUNT = 10000

TRAIN_WM_METHODS = ["LSB", "QIM", "DSSS", "AudioSeal", "Timbre", "Phase"]
TEST_WM_METHODS = ["Patchwork", "Echo", "WavMark", "Perth"]

#TEST1_NO_BOX_PERTURBS = ['encodec','background_noise']

#TRAIN_BLACK_BOX_PERTURBS = ['HSJA_signal', 'HSJA_spectrogram']
TEST_BLACK_BOX_PERTURBS = ['square', 'HSJA_signal', 'HSJA_spectrogram']

TEST_NO_BOX_PERTURBS = ['time_stretch', 'encodec','background_noise']

CHINESE_FILES = {"zh-CN", "zh-HK", "zh-TW", "yue"}
ENGLISH_FILES = {"en"}

def collect_files(root):
    
    audio_files = []
    for r, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".wav"):
                audio_files.append(os.path.join(r, n))
    return sorted(audio_files)

def parse_commonvoice_lang(path):
    base = os.path.basename(path)
    parts = base.split("_")
    
    if parts[1] in CHINESE_FILES or parts[1] in ENGLISH_FILES:
        return "en_or_cn"
    elif "-" in parts[1] or parts[1].isalpha():
        return "other"
    return "unknown"

def pick_n(files, n):
    if len(files) >= n:
        return files[:n]
    print(f"Warning: requested {n} but only {len(files)} available")
    return files

def assign_one_wm_per_file(file_list, wm_methods):
    n = len(file_list)
    per = n // len(wm_methods)
    remainder = n % len(wm_methods)
    wm_assign = []
    for i, m in enumerate(wm_methods):
        count = per + (1 if i < remainder else 0)
        wm_assign += [m] * count
    random.shuffle(wm_assign)
    return dict(zip(file_list, wm_assign))

def find_watermarked_file(clean_file_path, wm_method):
   
    clean_dir = os.path.dirname(clean_file_path)
    clean_basename = os.path.basename(clean_file_path)
    clean_name = os.path.splitext(clean_basename)[0]
    
    wm_dir = os.path.join(os.path.dirname(clean_dir), wm_method)
    
    if not os.path.exists(wm_dir):
        print(f"Warning: directory not found: {wm_dir}")
        return None
    
    wm_pattern = os.path.join(wm_dir, f"*{clean_name}*")
    wm_files = glob(wm_pattern)
    
    if wm_files:
        return wm_files[0]
    else:
        print(f"No watermarked file found for {clean_file_path} in {wm_method}")
        return None

def add_clean_and_watermarked(pool, split, wm_methods, manifest):
   
    wm_assign = assign_one_wm_per_file(pool, wm_methods)
    for clean_file in pool:
        wm_method = wm_assign[clean_file]
       
        # Add clean audio entry
        manifest.append({
            "orig_path": clean_file, 
            "split": split, 
            "dataset": split,
            "sampling_rate_khz": 16, 
            "is_watermarked": 0, 
            "watermark_method": "", 
            "perturbation": "", 
            "derived_path": clean_file
        })
        
       
        wm_file_path = find_watermarked_file(clean_file, wm_method)
        if wm_file_path:
            manifest.append({
                "orig_path": clean_file, 
                "split": split, 
                "dataset": split,
                "sampling_rate_khz": 16, 
                "is_watermarked": 1, 
                "watermark_method": wm_method, 
                "perturbation": "", 
                "derived_path": wm_file_path
            })
        else:
            print(f"Warning: Skipping watermarked entry for {clean_file}")

def add_no_box_perturbation(manifest):

    originals = [e for e in manifest if e.get("perturbation","") == ""]

    print(f"[INFO] Found {len(originals)} original files for no-box augmentation")

    for entry in originals:

        orig_wav = entry["derived_path"]
        split = entry["split"]
        is_wm = int(entry["is_watermarked"])
        wm_method = entry.get("watermark_method", "")

        base = os.path.splitext(os.path.basename(orig_wav))[0]
        folder = os.path.dirname(orig_wav)

        output_root = os.path.join(folder, "no_box_perturbations")

        if split in ["train", "validation"]:
            continue
        else:
            ptypes = TEST_NO_BOX_PERTURBS

        for p in ptypes:
            deprived_wav_path = apply_no_box_pert(
                input_wav=orig_wav,
                output_dir=output_root,
                common_perturbation=p
            )
            
            manifest.append({
                "orig_path": entry["orig_path"], 
                "split": split, 
                "dataset": split,
                "sampling_rate_khz": 16, 
                "is_watermarked": is_wm, 
                "watermark_method": wm_method if is_wm else "", 
                "perturbation": p, 
                "derived_path": deprived_wav_path
            })

def is_original_watermarked(e):
    return (int(e.get("is_watermarked", 0)) == 1 and e.get("perturbation", "") == "")

def is_original_unwatermarked(e):
    return (int(e.get("is_watermarked", 0)) == 0 and e.get("perturbation", "") == "")


def add_black_box_perturbation(manifest):

    ALLOWED_WM_METHODS = {
    "wavmark",
    "patchwork",
    "echo",
    }
    wm_entries = [e for e in manifest if is_original_watermarked(e) and e.get("watermark_method", "").lower() in ALLOWED_WM_METHODS]

    if len(wm_entries) == 0:
        print("no original watermarked files found")
        return
    
    wm_by_method = defaultdict(list)
    for e in wm_entries:
        wm_by_method[e["watermark_method"]].append(e)

    for wm_method, rows in wm_by_method.items():
        if len(rows) == 0:
            continue

        selected = random.sample(rows, min(200, len(rows)))
        print(f"[INFO] {wm_method}: selected {len(selected)} files for black-box attacks")

        for entry in selected:
            orig_wav = entry["derived_path"]
            split = entry["split"]
            base = os.path.basename(orig_wav)
            folder = os.path.dirname(orig_wav)
            
            output_root = os.path.join(folder, "black_box_perturbations")
            if split in ["train", "validation"]:
                continue
            else:
                ptypes = TEST_BLACK_BOX_PERTURBS

            for p in ptypes:
                output = os.path.join(output_root, p)

                if p == "HSJA_signal" and wm_method.lower() not in ["dsss", "phase"] :

                    subprocess.run(
                      [
                          "python3",
                          "black-box-HSJA_signal.py",
                          "--gpu", "0",
                          "--input_dir", orig_wav,
                          "--testset_size", str(len(selected)),
                          "--query_budget", "10000",
                          "--tau", "0.15",
                          "--norm", "linf",
                          "--model", wm_method.lower(),
                          "--blackbox_folder", output_root,
                          ],
                          check = True
                      
                    )
                      
                elif p == "HSJA_spectrogram" :
                    subprocess.run(
                        [
                            "python3",
                            "black-box-HSJA_spectrogram.py",
                            "--gpu", "0",
                            "--input_dir", orig_wav,
                            "--testset_size", str(len(selected)),
                            "--query_budget", "10000",
                            "--tau", "0.15", 
                            "--norm", "linf", 
                            "--model", wm_method.lower(),
                            "--blackbox_folder", output_root,
                            "--attack_type" ,"both",
                            ],
                            check = True
                    )
                elif p == "square" and wm_method.lower() not in ["dsss"]:
                    subprocess.run(
                        [
                            "python3", 
                            "black-box_square.py", 
                            "--gpu", "0", 
                            "--input_dir", orig_wav,
                            "--testset_size", str(len(selected)),  
                            "--query_budget", "10000"  
                            "--tau", "0.15", 
                            "--model", wm_method.lower(),
                            "--blackbox_folder", output_root,
                            "--attack_type", "both", 
                            "--eps", "0.05",
                        ],
                        check=True
                    )
                manifest.append({
                    "orig_path": entry["orig_path"], 
                    "split": split, 
                    "dataset": split,
                    "sampling_rate_khz": 16, 
                    "is_watermarked": 1, 
                    "watermark_method": wm_method, 
                    "perturbation": p, 
                    "derived_path": os.path.join(output, base)
                })

def add_white_box_perturbation(manifest):

    ALLOWED_WM_METHODS = {
    "wavmark",
    "patchwork",
    "echo",
    }
    wm_entries_removal = [e for e in manifest if is_original_watermarked(e) and e.get("watermark_method", "").lower() in ALLOWED_WM_METHODS and e.get("split", "") in ["test1_in", "test2_in"]]    # the ones with no pert and they are not watermarked, correct this

    if len(wm_entries_removal) == 0:
        print("no original watermarked files found")
        return
    
    wm_by_method = defaultdict(list)
    for e in wm_entries_removal:
        wm_by_method[e["watermark_method"]].append(e)

    for wm_method, rows in wm_by_method.items():
        if len(rows) == 0:
            continue

        selected = random.sample(rows, min(200, len(rows)))
        print(f"[INFO] {wm_method}: selected {len(selected)} files for white-box attacks")

        for entry in selected:
            orig_wav = entry["derived_path"]
            split = entry["split"]
            base = os.path.basename(orig_wav)
            folder = os.path.dirname(orig_wav)
            
            output_root = os.path.join(folder, "white_box_perturbations")
            

                
            output = os.path.join(output_root, "whitebox_removal")

            

            subprocess.run(
                [
                    "python3",
                    "white-box_removal.py",
                    "--gpu", "0",
                    "--input_dir", orig_wav,
                    "--tau", "0.1",
                    "--model", wm_method.lower(),
                    "--iter", "10000",
                    "--rescale_snr", "20",
                    "--whitebox_folder", output_root,
                ],
                check = True
            
            )
                
                    
                
            manifest.append({
                "orig_path": entry["orig_path"], 
                "split": split, 
                "dataset": split,
                "sampling_rate_khz": 16, 
                "is_watermarked": 1, 
                "watermark_method": wm_method, 
                "perturbation": "whitebox_removal", 
                "derived_path": os.path.join(output, base)
            })
    entries_forgery = [e for e in manifest if is_original_unwatermarked(e) and e.get("split", "") in ["test1_in", "test2_in"]]   
    

    if len(entries_forgery) == 0:
        print("[WARN] No clean unwatermarked files found for forgery attacks")
        return

    
    
    for wm_method in ALLOWED_WM_METHODS:

        if wm_method.lower() in TEST_WM_METHODS:
            selected = random.sample(entries_forgery, min(200, len(entries_forgery)))
        
            if len(selected) == 0:
                print(f"[WARN] {wm_method}: insufficient clean files for forgery")
                continue
            
        
            for entry in selected:
                orig_wav = entry["derived_path"]
                split = entry["split"]
                base = os.path.basename(orig_wav)
                folder = os.path.dirname(orig_wav)
                
                output_root = os.path.join(folder, "white_box_perturbations")
                output = os.path.join(output_root, "whitebox_forgery")
                
                subprocess.run(
                    [
                        "python3",
                        "white-box_forgery.py",
                        "--gpu", "0",
                        "--input_dir", orig_wav,
                        "--tau", "0.1",
                        "--model", wm_method.lower(),
                        "--iter", "10000",
                        "--rescale_snr", "20",
                        "--whitebox_folder", output_root,
                    ],
                    check = True
                
                )
                
                
                manifest.append({
                    "orig_path": entry["orig_path"], 
                    "split": split, 
                    "dataset": split,
                    "sampling_rate_khz": 16, 
                    "is_watermarked": 0, 
                    "watermark_method": "", 
                    "perturbation": "whitebox_forgery", 
                    "derived_path": os.path.join(output, base)
                })
        
        




def main():
    random.seed(42)

    CLEAN_ROOTS = {
        "LibriSpeech": "path to clean librispeech",
        "CommonVoice": "path to clean commonvoice", 
        "AISHELL": "path to clean aishell",
        "VCTK": "path to clean vctk"
    }

    print("Collecting datasets...")
    pools = {name: collect_files(path) for name, path in CLEAN_ROOTS.items()}
    for k, v in pools.items():
        print(f" - {k}: {len(v)} files")

    cv_en_cn = [f for f in pools["CommonVoice"] if parse_commonvoice_lang(f) == "en_or_cn"]
    cv_other = [f for f in pools["CommonVoice"] if parse_commonvoice_lang(f) == "other"]

    print(f"CommonVoice English/Chinese: {len(cv_en_cn)}, other languages: {len(cv_other)}")

    train_pool = pick_n(pools["LibriSpeech"], 20000) + pick_n(cv_en_cn, 4000) + pick_n(pools["AISHELL"], 10000)
    random.shuffle(train_pool)
    train_pool = train_pool[:TRAIN_CLEAN_COUNT]

    val_count = int(len(train_pool) * VAL_RATIO)
    val_pool = train_pool[:val_count]
    train_pool = train_pool[val_count:]

    test1_in_pool = pick_n(cv_other, 16000) 
    test1_in_pool = test1_in_pool[:TEST1_COUNT]

    test2_in_pool = pick_n(pools["VCTK"], 10000)
    test2_in_pool = test2_in_pool[:TEST2_COUNT]

    print(f"Train={len(train_pool)}, Val={len(val_pool)}, Test1_in={len(test1_in_pool)}, Test2_in={len(test2_in_pool)}")

    manifest = []
    fields = [
        "orig_path", "split", "dataset",
        "sampling_rate_khz", "is_watermarked",
        "watermark_method", "perturbation", "derived_path"
    ]

    add_clean_and_watermarked(train_pool, "train", TRAIN_WM_METHODS, manifest)
    add_clean_and_watermarked(val_pool, "validation", TRAIN_WM_METHODS, manifest)
    add_clean_and_watermarked(test1_in_pool, "test1_in", TEST_WM_METHODS, manifest)
    add_clean_and_watermarked(test2_in_pool, "test2_in", TEST_WM_METHODS, manifest)
    add_no_box_perturbation(manifest)
    add_black_box_perturbation(manifest)
    add_white_box_perturbation(manifest)

    output_file = "dataset_manifest.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            # Ensure all fields are present
            for field in fields:
                if field not in row:
                    row[field] = ""
            writer.writerow(row)
    
    print(f"Manifest saved to {output_file}")
    print(f"Total entries: {len(manifest)}")

if __name__ == "__main__":
    main()
