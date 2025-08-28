import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import soundfile as sf

def normalize_wav(wav: torch.Tensor) -> torch.Tensor:
    return wav / (wav.abs().max() + 1e-10)


def waveform_modification(filepath: str, pad_to: int, encoder):
    wav_np, sr = sf.read(filepath)      
    if len(wav_np) > pad_to:
        wav_np = wav_np[:pad_to]

    wav = torch.from_numpy(wav_np).float()
    wav = normalize_wav(wav)

    pad_mask = torch.zeros(encoder.n_frames, dtype=torch.bool)
    return wav, pad_mask
    
class WeaklyLabeledDataset(Dataset):
    def __init__(self, dataset_dir: str, labels: list, return_name: bool, encoder):
        self.dataset_dir = dataset_dir
        self.labels = labels
        self.lab2idx = {c: i for i, c in enumerate(labels)}
        self.encoder = encoder
        self.pad_to = int(encoder.audio_len * encoder.sr) 
        self.return_name = return_name

        self.items = []
        for cls in labels:
            cdir = os.path.join(dataset_dir, cls)
            if not os.path.isdir(cdir):
                continue
            for rec in os.listdir(cdir):
                rdir = os.path.join(cdir, rec)
                if not os.path.isdir(rdir):
                    continue
                for fn in os.listdir(rdir):
                    if fn.lower().endswith(".wav"):
                        self.items.append((os.path.join(rdir, fn), cls))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cls = self.items[idx]
        wav_np, sr = sf.read(path)            
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(-1)          
            
        if len(wav_np) > self.pad_to:
            wav_np = wav_np[:self.pad_to]
        wav = torch.from_numpy(wav_np).float() # [T]
        wav = normalize_wav(wav)
        
        n_frames = self.encoder.n_frames
        y = torch.zeros(len(self.labels), n_frames)
        y[self.lab2idx[cls]] = 1.0

        pad_mask = torch.zeros(n_frames, dtype=torch.bool)

        out_args = [wav, y, pad_mask, idx]
        if self.return_name:
            out_args.extend([os.path.basename(path), path])
        return out_args
           
    
def setmelspectrogram(feature_cfg):
    return torchaudio.transforms.MelSpectrogram(sample_rate=feature_cfg["sample_rate"],
                                                n_fft=feature_cfg["n_window"],
                                                win_length=feature_cfg["n_window"],
                                                hop_length=feature_cfg["hop_length"],
                                                f_min=feature_cfg["f_min"],
                                                f_max=feature_cfg["f_max"],
                                                n_mels=feature_cfg["n_mels"],
                                                window_fn=torch.hamming_window,
                                                wkwargs={"periodic": False},
                                                power=1) # 1:energy, 2:power
