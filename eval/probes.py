"""
Linear Probes

Evaluate what information is encoded in each layer's representations.
Must match LeWM's probe format for direct comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import pearsonr


class LinearProbe(nn.Module):
    """Linear probe for evaluating representation quality.

    Trains a single linear layer from frozen representations to predict
    a target quantity. Reports Pearson-r correlation.
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def train_probe(
    embeddings: Tensor,
    targets: Tensor,
    input_dim: int | None = None,
    output_dim: int = 1,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, float]:
    """Train a linear probe and return metrics.

    Args:
        embeddings: (N, D) frozen representations
        targets: (N, O) target values
        input_dim: override input dimension
        output_dim: target dimension
        epochs: training epochs
        lr: learning rate
        batch_size: batch size
        device: device

    Returns:
        dict with 'pearson_r', 'mse' for each output dimension
    """
    N = embeddings.shape[0]
    D = input_dim or embeddings.shape[-1]
    embeddings = embeddings.view(N, -1)[:, :D].to(device)
    targets = targets.view(N, -1)[:, :output_dim].to(device)

    # Train/val split (80/20)
    split = int(0.8 * N)
    train_emb, val_emb = embeddings[:split], embeddings[split:]
    train_tgt, val_tgt = targets[:split], targets[split:]

    train_ds = TensorDataset(train_emb, train_tgt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe = LinearProbe(D, output_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        probe.train()
        for emb_batch, tgt_batch in train_loader:
            pred = probe(emb_batch)
            loss = F.mse_loss(pred, tgt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        val_pred = probe(val_emb).cpu().numpy()
        val_true = val_tgt.cpu().numpy()

    results = {}
    results["mse"] = float(np.mean((val_pred - val_true) ** 2))

    # Pearson-r per output dimension
    r_values = []
    for d in range(output_dim):
        r, _ = pearsonr(val_pred[:, d], val_true[:, d])
        r_values.append(r)
    results["pearson_r"] = float(np.mean(r_values))
    results["pearson_r_per_dim"] = r_values

    return results


# Standard probe targets matching LeWM's evaluation
PROBE_TARGETS = {
    "agent_location": {"output_dim": 2, "description": "Agent (x, y) position"},
    "block_location": {"output_dim": 2, "description": "Block (x, y) position"},
    "block_angle": {"output_dim": 1, "description": "Block rotation angle"},
    "mass_proxy": {"output_dim": 1, "description": "Object mass (hidden property)"},
    "friction_proxy": {"output_dim": 1, "description": "Object friction (hidden property)"},
}
