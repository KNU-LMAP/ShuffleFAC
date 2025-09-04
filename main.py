import os, csv, random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import yaml
from tqdm import tqdm
from codecarbon import OfflineEmissionsTracker

from utils.model import CRNN
from utils.data_preprocessing import dataset
from utils.utils import calculate_macs, count_parameters

# ---------------------
# Reproducibility
# ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional strict determinism (can slow down, but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def kd_train(student, teacher, train_loader, optimizer, criterion, loss_fn_distill, device):
    student.train()
    teacher.eval()
    train_loss = 0.0
    num_batches = 0
    alpha = 0.7
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_outputs = teacher(batch_x)
        student_outputs = student(batch_x)
        loss_student = criterion(student_outputs, batch_y)

        loss_distill = loss_fn_distill(
            F.log_softmax(student_outputs / 4, dim=1),
            F.softmax(teacher_outputs / 4, dim=1)
        )

        total_loss = alpha * loss_distill + (1 - alpha) * loss_student
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        num_batches += 1
        
    return train_loss / max(1, num_batches)

def train(student, train_loader, optimizer, criterion, device):
    student.train()
    loss_student = 0.0
    num_batches = 0

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        student_outputs = student(batch_x)
        loss_student = criterion(student_outputs, batch_y)

        loss_student.backward()
        optimizer.step()
        loss_student += loss_student.item()
        num_batches += 1
        
    return loss_student / max(1, num_batches)

@torch.no_grad()
def valid(student, val_loader, criterion, device):
    student.eval()
    val_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(val_loader, total=len(val_loader), desc='Valid', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        num_batches += 1
        pred = torch.argmax(outputs, dim=1)
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if len(y_true_all) == 0:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    val_acc = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average='macro')

    return (val_loss / max(1, num_batches)), val_acc, val_macro_f1

@torch.no_grad()
def test(student, test_loader, criterion, device):
    student.eval()
    test_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(test_loader, total=len(test_loader), desc='Test', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        num_batches += 1
        pred = torch.argmax(outputs, dim=1)
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if len(y_true_all) == 0:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return (test_loss / max(1, num_batches)), test_acc, test_macro_f1


def main():
    set_seed(42)

    with open('/home/user/Deepship/default.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    crnn_cfg = configs["CRNN"]
    teacher_cfg = configs["teacher"]
    # IMPORTANT: make sure this matches your preprocessing output root
    DATA_ROOT = "/home/user/Deepship/preprocessed_data"
    train_set = dataset(os.path.join(DATA_ROOT, "train"))
    val_set   = dataset(os.path.join(DATA_ROOT, "val"))
    test_set  = dataset(os.path.join(DATA_ROOT, "test"))

    print(f"train/val/test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}", flush=True)
    if len(train_set) == 0:
        print("[WARN] Dataset is empty. Check path "
              f"'{DATA_ROOT}' and class folder names.", flush=True)

    # Dataloaders
    num_workers = 4
    train_loader = DataLoader(train_set, batch_size=48, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=48, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=48, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    student = CRNN(**crnn_cfg)
    teacher = CRNN(**teacher_cfg)
    checkpoint = torch.load('/home/user/Deepship/checkpoints/best_teacher.pt')
    teacher.load_state_dict(checkpoint['model_state'], strict=False)
    macs, _ = calculate_macs(student, configs)
    total_params, trainable_params = count_parameters(student)

    print("---------------------------------------------------------------")
    print("Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"MACs: {macs}")
    print("---------------------------------------------------------------\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = student.to(device)
    teacher = teacher.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss_fn_distill = nn.KLDivLoss(reduction='batchmean')

    # Logging / checkpoints
    log_path = '/home/user/Deepship/training_log.csv'
    ckpt_dir = '/home/user/Deepship/checkpoints'
    exp_dir  = '/home/user/Deepship/exp'
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_f1 = -1.0
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1'])

    # Emissions tracker (covers train+val+test)
    os.makedirs(os.path.join(exp_dir, "devtest_codecarbon"), exist_ok=True)
    tracker = OfflineEmissionsTracker(
        "DCASE Task 4 SED EXP",
        output_dir=os.path.join(exp_dir, "devtest_codecarbon"),
        log_level="warning",
        country_iso_code="KOR",  # set to your actual location; was "FRA"
    )
    tracker.start()

    num_epochs = 200
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}] start", flush=True)
        train_loss = train(student,train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = valid(student, val_loader, criterion, device)

        print(f"epoch {epoch+1}: "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f}")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"])

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {'epoch': epoch + 1, 'model_state': student.state_dict(), 'best_f1': best_f1},
                os.path.join(ckpt_dir, 'best.pt')
            )

    # Load best checkpoint for testing
    best_ckpt = os.path.join(ckpt_dir, 'best.pt')
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        student.load_state_dict(state['model_state'])
        print(f"Loaded best checkpoint (epoch={state.get('epoch')}, best_f1={state.get('best_f1'):.4f})")

    test_loss, test_acc, test_macro_f1 = test(student, test_loader, criterion, device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")

    emissions = tracker.stop()
    print(f"[CodeCarbon] Estimated emissions: {emissions} kg CO2eq")

if __name__ == "__main__":
    main()
