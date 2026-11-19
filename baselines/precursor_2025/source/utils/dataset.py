import torch
from torch.utils.data import Dataset

class ScenarioDataset(Dataset):
    def __init__(self, scenarios):
        self.scenarios = scenarios
    def __len__(self): return len(self.scenarios)
    def __getitem__(self, idx): return self.scenarios[idx]

def collate_fn(batch):
    """
    Produces:
      X:   (B, S_max, F) float tensor (already on device)
      mask:(B, S_max) bool tensor, True for valid elements (alternatives)
      prefs:(B, S_max) float tensor, NaN where preference is missing/unlabeled
      confs:(B, S_max) float tensor, NaN where confidence is missing/unlabeled
      groups:(B,) int tensor, 0=control, 1=non-control, 2=expert
    """

    sizes = [s['features'].shape[0] for s in batch]
    max_n = max(sizes)
    feat_dim = batch[0]['features'].shape[1]
    device = batch[0]['features'].device  

    X = torch.zeros(len(batch), max_n, feat_dim, dtype=torch.float, device=device)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool, device=device)
    prefs = torch.full((len(batch), max_n), float('nan'), dtype=torch.float, device=device)
    confs = torch.full((len(batch), max_n), float('nan'), dtype=torch.float, device=device)
    groups = torch.zeros(len(batch), dtype=torch.int, device=device)  # 0=control, 1=non-control #2=expert

    for i, s in enumerate(batch):
        n = s['features'].shape[0]
        X[i, :n, :] = s['features']  
        mask[i, :n] = True

        prefs[i, :n] = s['prefs']
        confs[i, :n] = s['confs']

        if str(s['id']).startswith('control'):
            groups[i] = 0
        elif str(s['id']).startswith('expert'):
            groups[i] = 2
        else:
            groups[i] = 1

        

    return X, mask, prefs, confs, groups
