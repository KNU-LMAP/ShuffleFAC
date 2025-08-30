from __future__ import annotations
import os
import yaml
import logging
import shutil
from glob import glob
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelF1Score

from utils.utils import Encoder, Scaler
from utils.model import CRNN
from utils.dataset import WeaklyLabeledDataset


def get_configs(config_dir, server_config_dir="./configs/config_server.yaml"):
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)
    with open(server_config_dir, "r") as f:
        server_cfg = yaml.safe_load(f)
    train_cfg = configs["training"]
    feature_cfg = configs["feature"]
    train_cfg["batch_sizes"] = server_cfg.get("batch_size", 32)
    train_cfg["net_pooling"] = feature_cfg["net_subsample"]
    return configs, server_cfg, train_cfg, feature_cfg


def get_save_directories(configs, train_cfg, iteration, args):
    general_cfg = configs["generals"]
    save_folder = general_cfg["save_folder"]
    savepsds = general_cfg.get("savepsds", False)
    if "new_exp" in save_folder:
        save_folder = save_folder + f'_gpu={args.gpu}'
        configs["generals"]["save_folder"] = save_folder
    if not train_cfg.get("test_only", False):
        if iteration is not None:
            save_folder = save_folder + f'_iter_{iteration}'
            configs["generals"]["save_folder"] = save_folder
        print("save directory : " + save_folder)
        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
            yaml.dump(configs, f)
    stud_best_path = os.path.join(save_folder, "best_student.pt")
    tch_best_path = os.path.join(save_folder, "best_teacher.pt")
    train_cfg["best_paths"] = [stud_best_path, tch_best_path]
    if savepsds:
        stud_psds_folder = os.path.join(save_folder, "psds_student")
        tch_psds_folder = os.path.join(save_folder, "psds_teacher")
        psds_folders = [stud_psds_folder, tch_psds_folder]
    else:
        psds_folders = [None, None]
    train_cfg["psds_folders"] = psds_folders
    return configs, train_cfg


def get_logger(save_folder):
    logger = logging.getLogger()
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_folder, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_labeldict():
    return OrderedDict({
        "cargo": 0,
        "passenger": 1,
        "tanker": 2,
        "tug": 3
    })


def get_encoder(LabelDict, feature_cfg, audio_len):
    return Encoder(list(LabelDict.keys()),
                   audio_len=audio_len,
                   frame_len=feature_cfg["frame_length"],
                   frame_hop=feature_cfg["hop_length"],
                   net_pooling=feature_cfg["net_subsample"],
                   sr=feature_cfg["sr"])


def _as_int(bs):
    if isinstance(bs, (list, tuple)):
        return int(bs[0])
    return int(bs)


def get_mt_datasets(configs, server_cfg, train_cfg):
    encoder = train_cfg["encoder"]
    dataset_cfg = configs["dataset"]
    ds_root = dataset_cfg["root"]
    labels = list(get_labeldict().keys())
    batch_size = _as_int(server_cfg.get("batch_size", 32))
    batch_size_val = int(server_cfg.get("batch_size_val", max(32, batch_size)))
    num_workers = int(server_cfg.get("num_workers", 4))
    train_dir = os.path.join(ds_root, "train")
    val_dir = os.path.join(ds_root, "val")
    test_dir = os.path.join(ds_root, "test")
    train_dataset = WeaklyLabeledDataset(train_dir, labels, False, encoder)
    valid_dataset = WeaklyLabeledDataset(val_dir, labels, True, encoder)
    test_dataset = WeaklyLabeledDataset(test_dir, labels, True, encoder)
    train_cfg["trainloader"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_cfg["validloader"] = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_cfg["testloader"] = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_cfg["train_tsvs"] = None
    train_cfg["valid_tsvs"] = None
    train_cfg["test_tsvs"] = None
    return train_cfg


def get_models(configs, train_cfg, multigpu=False):
    net = CRNN(**configs["CRNN"])
    ema_net = deepcopy(net)
    for p in ema_net.parameters():
        p.detach_()
    device = train_cfg.get("device", torch.device("cpu"))
    if multigpu and train_cfg.get("n_gpu", 1) > 1:
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)
    net = net.to(device)
    ema_net = ema_net.to(device)
    return net, ema_net


def get_scaler(scaler_cfg):
    return Scaler(statistic=scaler_cfg["statistic"],
                  normtype=scaler_cfg["normtype"],
                  dims=tuple(scaler_cfg["dims"]))


def get_f1calcs(n_class, device):
    stud_f1calc = MultilabelF1Score(num_labels=n_class, average="macro", threshold=0.5, zero_division=0)
    tch_f1calc = MultilabelF1Score(num_labels=n_class, average="macro", threshold=0.5, zero_division=0)
    return stud_f1calc.to(device), tch_f1calc.to(device)


def get_printings():
    printing_epoch = (
        '[Epc %d] tt: %0.3f, cl_st: %0.3f, cl_wk: %0.3f, '
        'cn_st: %0.3f, cn_wk: %0.3f, st_vl: %0.3f, t_vl: %0.3f, t: %ds'
    )
    printing_test = (
        "      test result is out!"
        "\n      [student] psds1: %.4f, psds2: %.4f"
        "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "
        "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"
        "\n      [teacher] psds1: %.4f, psds2: %.4f"
        "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "
        "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"
    )
    return printing_epoch, printing_test


class History:
    def __init__(self):
        self.history = {"train_total_loss": [], "train_class_strong_loss": [], "train_class_weak_loss": [],
                        "train_cons_strong_loss": [], "train_cons_weak_loss": [], "stud_val_metric": [],
                        "tch_val_metric": []}

    def update(self, train_return, val_return):
        total, class_str, class_wk, cons_str, cons_wk = train_return
        stud_val_metric, tch_val_metric = val_return
        self.history['train_total_loss'].append(total)
        self.history['train_class_strong_loss'].append(class_str)
        self.history['train_class_weak_loss'].append(class_wk)
        self.history['train_cons_strong_loss'].append(cons_str)
        self.history['train_cons_weak_loss'].append(cons_wk)
        self.history['stud_val_metric'].append(stud_val_metric)
        self.history['tch_val_metric'].append(tch_val_metric)
        return stud_val_metric, tch_val_metric

    def save(self, save_dir):
        import pickle
        with open(save_dir, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BestModels:
    def __init__(self):
        self.stud_best_val_metric = 0.0
        self.tch_best_val_metric = 0.0
        self.stud_best_state_dict = None
        self.tch_best_state_dict = None

    def update(self, train_cfg, logger, val_metrics):
        stud_update = False
        tch_update = False
        if val_metrics[0] > self.stud_best_val_metric:
            self.stud_best_val_metric = val_metrics[0]
            self.stud_best_state_dict = train_cfg["net"].state_dict()
            stud_update = True
        if val_metrics[1] > self.tch_best_val_metric:
            self.tch_best_val_metric = val_metrics[1]
            self.tch_best_state_dict = train_cfg["ema_net"].state_dict()
            tch_update = True
        if train_cfg.get("epoch", 0) > int(train_cfg.get("n_epochs", 0) * 0.5):
            if stud_update and tch_update:
                logger.info(f"     best student & teacher model updated at epoch {train_cfg['epoch'] + 1}!")
            elif stud_update:
                logger.info(f"     best student model updated at epoch {train_cfg['epoch'] + 1}!")
            elif tch_update:
                logger.info(f"     best teacher model updated at epoch {train_cfg['epoch'] + 1}!")
        return logger

    def get_bests(self, best_paths):
        torch.save(self.stud_best_state_dict, best_paths[0])
        torch.save(self.tch_best_state_dict, best_paths[1])
        return self.stud_best_val_metric, self.tch_best_val_metric


def get_ensemble_models(train_cfg):
    ensemble_folder = train_cfg["ensemble_dir"]
    stud_nets_saved = glob(os.path.join(ensemble_folder, '*/best_student.pt'))
    tch_nets_saved = glob(os.path.join(ensemble_folder, '*/best_teacher.pt'))
    train_cfg["stud_nets"] = []
    train_cfg["tch_nets"] = []
    for p in stud_nets_saved:
        net_temp = deepcopy(train_cfg["net"]).to(train_cfg["device"])
        net_temp.load_state_dict(torch.load(p, map_location=train_cfg["device"]))
        train_cfg["stud_nets"].append(net_temp)
    for p in tch_nets_saved:
        net_temp = deepcopy(train_cfg["net"]).to(train_cfg["device"])
        net_temp.load_state_dict(torch.load(p, map_location=train_cfg["device"]))
        train_cfg["tch_nets"].append(net_temp)
    return train_cfg
