import os, random
from typing import Dict, List
import torchaudio as ta
import soundfile as sf
import shutil

# -------------------------------
# 설정값
# -------------------------------
RAW_ROOT = "/home/user/Deepship/data"                 # 원본 폴더 (class/REC/*.wav)
OUT_ROOT = "/home/user/Deepship/preprocessed_data_V2"  # 출력 루트
TMP_SEG_DIR = os.path.join(OUT_ROOT, "_segments")      # 임시 세그먼트 저장 위치
CLASSES  = ["Cargo", "Passengership", "Tanker", "Tug"]
SR       = 16000
WIN_SEC  = 3.0
HOP_SEC  = 3.0
SPLIT    = (0.7, 0.1, 0.2)
SEED     = 42

# -------------------------------------------------------
# 유틸 함수
# -------------------------------------------------------
def load_mono_resample(path: str, target_sr: int):
    wav, sr = ta.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = ta.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)

def save_wav(x, sr, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x.cpu().numpy(), sr)

def list_recordings(raw_root: str, classes: List[str]) -> Dict[str, List[str]]:
    out = {}
    for cls in classes:
        cdir = os.path.join(raw_root, cls)
        if not os.path.isdir(cdir):
            continue
        wavs = [os.path.join(cdir, fn) for fn in os.listdir(cdir) if fn.lower().endswith(".wav")]
        if wavs:
            out[cls] = wavs
    return out

# -------------------------------------------------------
# 세그먼트 생성
# -------------------------------------------------------
def segment_and_save_wav(in_path: str, out_dir: str, sr: int, win: int, hop: int, cls_name: str) -> int:
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_cls_dir = os.path.join(out_dir, cls_name)
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
        out_path = os.path.join(out_cls_dir, f"{base}_seg{i+1:04d}.wav")
        save_wav(seg, sr, out_path)
        total += 1
    return total

# -------------------------------------------------------
# 메인
# -------------------------------------------------------
def main():
    random.seed(SEED)
    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)

    # 1️⃣ 세그먼트 먼저 생성
    cls2recs = list_recordings(RAW_ROOT, CLASSES)
    if not cls2recs:
        print("[ERR] No recordings found. Check RAW_ROOT/CLASSES path.")
        return

    print("[STEP 1] Segmenting all audio files...")
    total_segments = 0
    for cls, recs in cls2recs.items():
        cnt_cls = 0
        for in_path in recs:
            n = segment_and_save_wav(in_path, TMP_SEG_DIR, SR, win, hop, cls)
            cnt_cls += n
        print(f"[SEG] {cls}: {cnt_cls} segments")
        total_segments += cnt_cls
    print(f"[INFO] Total segments generated: {total_segments}")

    # 2️⃣ 세그먼트를 train/val/test로 나누기
    print("\n[STEP 2] Splitting segments into train/val/test...")
    for cls in CLASSES:
        cls_dir = os.path.join(TMP_SEG_DIR, cls)
        if not os.path.exists(cls_dir):
            continue

        seg_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".wav")]
        random.shuffle(seg_files)

        n = len(seg_files)
        n_train = int(n * SPLIT[0])
        n_val   = int(n * SPLIT[1])
        n_test  = n - n_train - n_val

        splits = {
            "train": seg_files[:n_train],
            "val": seg_files[n_train:n_train+n_val],
            "test": seg_files[n_train+n_val:]
        }

        for sp, files in splits.items():
            out_cls_dir = os.path.join(OUT_ROOT, sp, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            for f in files:
                shutil.move(f, os.path.join(out_cls_dir, os.path.basename(f)))

        print(f"[SPLIT] {cls}: {n_train}/{n_val}/{n_test} (train/val/test)")

    # 3️⃣ 임시 세그먼트 폴더 삭제
    shutil.rmtree(TMP_SEG_DIR, ignore_errors=True)
    print(f"\n[Done] Segmentation + Split complete!")
    print(f"Output root: {OUT_ROOT}")

if __name__ == "__main__":
    main()
