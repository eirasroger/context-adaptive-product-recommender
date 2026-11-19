import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ..utils.dataset import collate_fn

# -----------------------------
# Dataset and Collate Function
# -----------------------------
class ScenarioDataset(Dataset):
    def __init__(self, scenarios):
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        return self.scenarios[idx]




# -----------------------------
# Loss Functions
# -----------------------------
def masked_weighted_smoothl1(pred, target, confs, eps=1e-8, min_conf=0.01, max_conf=1.0):
    device = pred.device
    if pred.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    valid_mask = ~torch.isnan(target)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    target_safe = torch.nan_to_num(target, nan=0.0)

    if confs is None:
        weights = torch.ones_like(target_safe, device=device)
    else:
        weights = confs.clone().to(device)
        weights = torch.nan_to_num(weights, nan=1.0)

    weights = torch.clamp(weights, min=min_conf, max=max_conf)
    weights = weights * valid_mask.float()

    sum_w = weights.sum()
    count_valid = valid_mask.float().sum()
    if sum_w.item() == 0:
        weights = valid_mask.float()
        sum_w = weights.sum()

    weights = weights * (count_valid / (sum_w + eps))

    loss_elem = nn.SmoothL1Loss(reduction='none')(pred, target_safe)

    weighted = loss_elem * weights
    loss = weighted.sum() / (count_valid + eps)

    return loss




def group_weighted_loss(pred, target, confs, groups, weights=None, eps=1e-8):
    device = pred.device
    groups_tensor = groups.to(device)

    if weights is None:
        weights = {0: 2/5, 1: 2/5, 2: 1/5}  # default weights for control (0), llm (1), expert (2)

    group_losses = {}
    total_loss = 0.0

    for group_idx, group_name in zip([0, 1, 2], ['control', 'llm', 'expert']):
        idx = (groups_tensor == group_idx).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            pred_g = pred[idx]
            target_g = target[idx]
            confs_g = confs[idx] if confs is not None else None
            # Check actual dimensionality and non-empty tensors
            if pred_g.numel() > 0 and target_g.numel() > 0 and (confs_g is None or confs_g.numel() > 0):
                group_loss = masked_weighted_smoothl1(pred_g, target_g, confs_g, eps=eps)
            else:
                group_loss = torch.tensor(0.0, device=device)
            group_losses[group_name] = group_loss
            total_loss += weights.get(group_idx, 0.0) * group_loss
        else:
            group_losses[group_name] = torch.tensor(0.0, device=device)

    return total_loss, group_losses['control'], group_losses['llm'], group_losses['expert']



# -----------------------------
# Training Loop
# -----------------------------
def train(model, train_loader, val_loader, optimizer, epochs, scheduler=None, device='cuda', early_stopping_patience=5, early_stopping_min_delta=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}") 
    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improve = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_control_loss = 0.0
        total_llm_loss = 0.0
        total_expert_loss = 0.0
        for batch in train_loader:
            Xb, maskb, prefsb, confsb, groups = batch
            Xb, maskb = Xb.to(device), maskb.to(device)
            prefsb, confsb = prefsb.to(device), confsb.to(device)

            optimizer.zero_grad()
            pred = model(Xb, maskb)
            loss, control_loss, llm_loss, expert_loss = group_weighted_loss(pred, prefsb, confsb, groups)
            loss.backward()
            optimizer.step()

            total_loss         += loss.item()
            total_control_loss += control_loss.item()
            total_llm_loss     += llm_loss.item()
            total_expert_loss  += expert_loss.item()

        avg_train_loss      = total_loss / len(train_loader)
        avg_control_loss    = total_control_loss / len(train_loader)
        avg_llm_loss        = total_llm_loss / len(train_loader)
        avg_expert_loss     = total_expert_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Validation -----
        model.eval()
        val_loss_accum  = 0.0
        val_control_accum = 0.0
        val_llm_accum   = 0.0
        val_expert_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                Xb, maskb, prefsb, confsb, groups = batch
                Xb, maskb = Xb.to(device), maskb.to(device)
                prefsb, confsb = prefsb.to(device), confsb.to(device)

                pred = model(Xb, maskb)  
                val_loss, val_control_loss, val_llm_loss, val_expert_loss = group_weighted_loss(pred, prefsb, confsb, groups)

                val_loss_accum     += val_loss.item()
                val_control_accum  += val_control_loss.item()
                val_llm_accum      += val_llm_loss.item()
                val_expert_accum   += val_expert_loss.item()
                val_batches        += 1

        avg_val_loss         = val_loss_accum / val_batches
        avg_val_control_loss = val_control_accum / val_batches
        avg_val_llm_loss     = val_llm_accum / val_batches
        avg_val_expert_loss  = val_expert_accum / val_batches
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_val_loss)
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Train Control: {avg_control_loss:.12f} | Train LLM: {avg_llm_loss:.6f} | Train Expert: {avg_expert_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | Val Control: {avg_val_control_loss:.12f} | Val LLM: {avg_val_llm_loss:.6f} | Val Expert: {avg_val_expert_loss:.6f} | LR: {lr:.6f}")

        if avg_val_loss + early_stopping_min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs. Best Val Loss: {best_val_loss:.6f}")
            model.load_state_dict(best_model_state)
            break

    return train_losses, val_losses
