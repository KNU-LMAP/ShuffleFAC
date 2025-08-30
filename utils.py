# utils/utils.py  (Deepship + FAC, weak-only)

import math
import numpy as np
import torch
import torch.nn as nn
import torchaudio


class Encoder:

    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000):
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling

        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples / 2 / self.frame_hop) * 2 / self.net_pooling)

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_weak(self, events):
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        for event in events:
            labels[self.labels.index(event)] = 1
        return labels


class ExponentialWarmup(object):
    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        current = np.clip(self.step_num, 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        return float(np.exp(self.exponent * phase * phase))


def update_ema(net, ema_net, step, ema_factor):
    alpha = min(1 - 1 / step, ema_factor)
    for ema_params, params in zip(ema_net.parameters(), net.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return ema_net


class Scaler(nn.Module):
    def __init__(self, statistic="instance", normtype="minmax", dims=(0, 2), eps=1e-8):
        super(Scaler, self).__init__()
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(Scaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.statistic == "dataset":
            super(Scaler, self)._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

    def forward(self, input):
        if self.statistic == "dataset":
            if self.normtype == "mean":
                return input - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (input - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        elif self.statistic == "instance":
            if self.normtype == "mean":
                return input - torch.mean(input, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (input - torch.mean(input, self.dims, keepdim=True)) / (
                    torch.std(input, self.dims, keepdim=True) + self.eps
                )
            elif self.normtype == "minmax":
                return (input - torch.amin(input, dim=self.dims, keepdim=True)) / (
                    torch.amax(input, dim=self.dims, keepdim=True)
                    - torch.amin(input, dim=self.dims, keepdim=True) + self.eps
                )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


class AsymmetricalFocalLoss(nn.Module):
    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricalFocalLoss, self).__init__()
        self.gamma = gamma   # class balancing
        self.zeta = zeta     # active/inactive balancing

    def forward(self, pred, target):
        losses = - (
            ((1 - pred) ** self.gamma) * target * torch.clamp_min(torch.log(pred), -100) +
            (pred ** self.zeta) * (1 - target) * torch.clamp_min(torch.log(1 - pred), -100)
        )
        return torch.mean(losses)


def take_log(feature: torch.Tensor) -> torch.Tensor:

    amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
    amp2db.amin = 1e-5
    return amp2db(feature).clamp(min=-50, max=80)


def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total_params += parameter.numel()
    return total_params
