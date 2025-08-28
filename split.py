# split.py
import os, math, random
from typing import Dict, List, Tuple
import torchaudio as ta
import soundfile as sf

# -------------------------------
# 설정값
# -------------------------------
RAW_ROOT = "/home/l13/deepship/dataset"                 # 원본 폴더 (class/REC/*.wav)
OUT_ROOT = "/home/l13/deepship/Segments_5s_16k"  # 출력 루트
CLASSES  = ["cargo", "passenger", "tanker", "tug"]      # 존재하는 클래스만 자동 처리
SR       = 16000                                        # 타깃 샘플레이트
WIN_SEC  = 5.0                                          # 고정 세그먼트 길이(초)
HOP_SEC  = 5.0                                          # 슬라이싱 간격(=WIN이면 non-overlap)
SPLIT    = (0.7, 0.1, 0.2)                              # (train, val, test)
SEED     = 42                                           # 고정 시드



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
        recs = []
        for rec in sorted(os.listdir(cdir)):
            rdir = os.path.join(cdir, rec)
            if os.path.isdir(rdir):
                recs.append(rdir)
        if recs:
            out[cls] = recs
    return out

def split_by_recording(recs: List[str], split=(0.7, 0.1, 0.2)) -> Tuple[List[str], List[str], List[str]]:
    n = len(recs)
    n_train = int(round(n * split[0]))
    n_val   = int(round(n * split[1]))
    n_test  = n - n_train - n_val
    train = recs[:n_train]
    val   = recs[n_train:n_train+n_val]
    test  = recs[n_train+n_val:]
    assert len(train)+len(val)+len(test) == n
    return train, val, test

def segment_one_wav(in_path: str, sr: int, win: int, hop: int) -> List[Tuple[int, int]]:
    wav = load_mono_resample(in_path, sr)
    N = wav.numel()
    if N < win:
        return []
    n_full = (N - win) // hop + 1
    segments = []
    for i in range(n_full):
        s = i * hop
        e = s + win
        if e <= N:       
            segments.append((s, e))
    return segments

def segment_and_save_rec(rec_dir: str, out_split_dir: str, sr: int, win: int, hop: int, cls_name: str) -> int:
    rec_id = os.path.basename(rec_dir)
    out_rec = os.path.join(out_split_dir, cls_name, rec_id)
    total = 0
    for fn in sorted(os.listdir(rec_dir)):
        if not fn.lower().endswith(".wav"):
            continue
        in_path = os.path.join(rec_dir, fn)
        try:
            wav = load_mono_resample(in_path, sr)
        except Exception as e:
            print(f"[WARN] skip {in_path}: {e}")
            continue

        N = wav.numel()
        if N < win:
            continue

        n_full = (N - win) // hop + 1
        base = os.path.splitext(fn)[0]
        for i in range(n_full):
            s = i * hop
            e = s + win
            if e > N:   
                break
            seg = wav[s:e]
            out_path = os.path.join(out_rec, f"{base}_{cls_name}-seg_{i+1:04d}.wav")
            save_wav(seg, sr, out_path)
            total += 1
    return total

def main():
    random.seed(SEED)

    win = int(WIN_SEC * SR)
    hop = int(HOP_SEC * SR)

    # 1) 클래스별 REC 폴더 수집
    cls2recs = list_recordings(RAW_ROOT, CLASSES)
    if not cls2recs:
        print("[ERR] No recordings found. Check RAW_ROOT/CLASSES path.")
        return

    # 2) 클래스별로 REC 셔플 & split
    splits = {"train": {}, "val": {}, "test": {}}
    for cls, rec_list in cls2recs.items():
        recs = rec_list[:]          # copy
        random.shuffle(recs)
        tr, va, te = split_by_recording(recs, SPLIT)
        splits["train"][cls] = tr
        splits["val"][cls]   = va
        splits["test"][cls]  = te

    # 3) 세그먼트 생성 & 저장
    grand_total = 0
    for sp in ["train", "val", "test"]:
        out_split_dir = os.path.join(OUT_ROOT, sp)
        for cls, recs in splits[sp].items():
            cnt_cls = 0
            for rec_dir in recs:
                n = segment_and_save_rec(rec_dir, out_split_dir, SR, win, hop, cls)
                cnt_cls += n
            print(f"[{sp}] {cls}: wrote {cnt_cls} segments")
            grand_total += cnt_cls

    print(f"\n[Done] total segments written: {grand_total}")
    print(f"Output root: {OUT_ROOT}")

if __name__ == "__main__":
    main()
