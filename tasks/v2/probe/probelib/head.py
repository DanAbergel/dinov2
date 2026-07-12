"""The probe head (PyTorch) + its training loop.

Probe is a small nn.Module on top of the FROZEN 384-d CLS embedding:
  hidden=()        -> a single Linear (logistic regression)
  hidden=(256,128) -> an MLP (Linear -> ReLU -> ... -> 1)
train_probe fits it with Adam + BCE + backprop and returns test AUROC/Acc/F1.
The encoder stays frozen — only this head is trained.
"""

import numpy as np
import torch
import torch.nn as nn

from .encoder import DEVICE
from .metrics import accuracy, auroc, f1


class Probe(nn.Module):
    """hidden=() -> Linear(d_in, 1); hidden=(256,128) -> MLP with ReLU. Output = 1 logit."""

    def __init__(self, d_in, hidden=()):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)                  # (B,) logits


def train_probe(Xtr, ytr, Xte, yte, hidden=(), epochs=200, lr=1e-3, weight_decay=1e-4):
    """Train a Probe head with Adam + BCE + backprop on the frozen embeddings;
    return test AUROC/Acc/F1. Features standardized on TRAIN stats; class balance
    handled by BCE pos_weight (= #neg / #pos)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6             # standardize on train
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEVICE)
    xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)

    model = Probe(Xtr.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    npos = float(ytr.sum())
    pos_weight = torch.tensor([(len(ytr) - npos) / max(npos, 1.0)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr_t)               # forward
        loss.backward()                                 # backprop
        opt.step()

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(xte)).cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    return {"auc": auroc(yte, proba), "acc": accuracy(yte, pred), "f1": f1(yte, pred),
            "n_train": int(len(ytr)), "n_test": int(len(yte)), "pos_test": int(yte.sum())}
