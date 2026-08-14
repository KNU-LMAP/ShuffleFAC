import glob
import os, math, random
from typing import Dict, List, Tuple
import torchaudio as ta
import soundfile as sf

RAW_ROOT = "/home/user/Desktop/deepship_data/DeepShip" # Set Your RAW file root
OUT_ROOT = "/home/user/Desktop/deepship_data/DeepShip/overlap_3s_10k"  #  Set Your output root

CLASSES  = ["Cargo", "Passengership", "Tanker", "Tug"]      # Only process existing classes
SR       = 16000                                        # Target Sampling Rate
WIN_SEC  = 3.0                                          # Set Segments(second)
HOP_SEC  = 3.0                                         # Slicing interval(if WIN_SEC=HOP_SEC, non-overlap)
SPLIT    = (0.7, 0.1, 0.2)                              # (train, val, test)
SEED     = 42                                           # Fix Seed


def load_mono_resample(path: str, target_sr: int):
    wav, sr = ta.load(path)           # [ch, T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = ta.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)             # [T]

def save_wav(x, sr, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x.cpu().numpy(), sr)

def list_recordings(raw_root: str, classes: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cls in classes:
        cdir = os.path.join(raw_root, cls)
        if not os.path.isdir(cdir):
            continue
        wavs = glob.glob(os.path.join(cdir, "**", "*.wav"), recursive=True)
        if wavs:
            out[cls] = wavs
    return out



def split_by_recording(recs: List[str], split=(0.6, 0.2, 0.2)) -> Tuple[List[str], List[str], List[str]]:
    n = len(recs)
    n_train = int(round(n * split[0]))
    n_val   = int(round(n * split[1]))
    n_test  = n - n_train - n_val
    train = recs[:n_train]
    val   = recs[n_train:n_train+n_val]
    test  = recs[n_train+n_val:]
    assert len(train)+len(val)+len(test) == n
    return train, val, test

def segment_and_save_wav(in_path: str, out_split_dir: str, sr: int, win: int, hop: int, cls_name: str) -> int:
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_cls_dir = os.path.join(out_split_dir, cls_name)
    os.makedirs(out_cls_dir, exist_ok=True)
    
    try:
        wav = load_mono_resample(in_path, sr)
    except Exception as e:
        print(f"[WARN] skip {in_path}: {e}")
        return 0

    N = wav.numel()
    if N < win:
        return 0

    n_full = (N - win) // hop + 1
    total = 0
    for i in range(n_full):
        s = i * hop
        e = s + win
        if e > N:
            break
        seg = wav[s:e]
        out_path = os.path.join(out_cls_dir, f"{base}_{cls_name}-seg_{i+1:04d}.wav")
        save_wav(seg, sr, out_path)
        total += 1
    return total


def main():
    random.seed(SEED)

    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)

    # 1) Collect wav file by classes
    cls2recs = list_recordings(RAW_ROOT, CLASSES)
    if not cls2recs:
        print("[ERR] No recordings found. Check RAW_ROOT/CLASSES path.")
        return

    # 2) wav file shuffle & splits
    splits = {"train": {}, "val": {}, "test": {}}
    for cls, rec_list in cls2recs.items():
        recs = rec_list[:]  # copy
        random.shuffle(recs)
        tr, va, te = split_by_recording(recs, SPLIT)
        splits["train"][cls] = tr
        splits["val"][cls]   = va
        splits["test"][cls]  = te

    # 3) Generate segment & save
    grand_total = 0
    for sp in ["train", "val", "test"]:
        out_split_dir = os.path.join(OUT_ROOT, sp)
        for cls, wavs in splits[sp].items():
            cnt_cls = 0
            for in_path in wavs:
                n = segment_and_save_wav(in_path, out_split_dir, SR, win, hop, cls)
                cnt_cls += n
            print(f"[{sp}] {cls}: wrote {cnt_cls} segments")
            grand_total += cnt_cls

    print(f"\n[Done] total segments written: {grand_total}")
    print(f"Output root: {OUT_ROOT}")

if __name__ == "__main__":
    main()
