"""Linear probe of a frozen encoder (protocol of Peng et al., 2022, footnote 1)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from multimodal_ml.metrics import compute_regression_metrics
from multimodal_ml.seed import set_seed

PROBE = {"epochs": 200, "lr": 1e-3, "patience": 10, "batch_size": 16}


@torch.no_grad()
def features(encoder, loader, device):
    encoder.eval()
    xs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        xs.append(encoder.encode(batch).cpu())
        ys.append(batch.labels.cpu())
    return torch.cat(xs), torch.cat(ys)


def probe(encoder, loaders, seed: int, device, config) -> dict[str, float]:
    """Freeze `encoder`; train only a new Linear(feature_dim, 1); test metrics of the best-val probe."""
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (features(encoder, l, device) for l in loaders)
    set_seed(seed)
    head = nn.Linear(x_tr.size(1), 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=PROBE["lr"])
    best, best_state, waited = float("inf"), None, 0
    for _ in range(PROBE["epochs"]):
        for idx in torch.randperm(len(x_tr)).split(PROBE["batch_size"]):
            loss = F.mse_loss(head(x_tr[idx]).squeeze(-1), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_mae = (head(x_va).squeeze(-1) - y_va).abs().mean().item()
        if val_mae < best:
            best, best_state, waited = val_mae, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PROBE["patience"]:
                break
    head.load_state_dict(best_state)
    with torch.no_grad():
        preds = head(x_te).squeeze(-1).numpy()
    return compute_regression_metrics(preds, y_te.numpy(), config)
