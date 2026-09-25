import os
import time
import torch
import sys
from pathlib import Path

repo_root = Path(".").resolve()
sys.path.insert(0, str(repo_root / "backend"))

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import MaskedLoss

print(f"os.cpu_count(): {os.cpu_count()}")
print(f"torch.get_num_threads(): {torch.get_num_threads()}")

config = STGNNConfig(
    input_dim=8,
    output_dim=4,
    hidden_dim=64,
    num_spatial_layers=2,
    num_temporal_layers=1,
    input_sequence_length=12,
    forecast_horizons=["PLUS_5MIN", "PLUS_10MIN", "PLUS_15MIN"],
    dropout=0.1,
)

for threads in [4, 8, 12, 16]:
    if threads <= os.cpu_count():
        torch.set_num_threads(threads)
        model = SpatioTemporalGNN(config)
        loss_fn = MaskedLoss(loss_type="smooth_l1")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        adj = torch.rand((170, 170))

        B, T, N, F = 32, 12, 170, 8
        H, T_target = 3, 4

        # warmup
        x = torch.randn((B, T, N, F))
        y = torch.randn((B, H, N, T_target))
        mask = torch.ones((B, H, N, T_target), dtype=torch.bool)
        pred = model(x, adj)
        loss = loss_fn(pred, y, mask=mask)
        loss.backward()
        optimizer.step()

        t0 = time.time()
        for i in range(5):
            x = torch.randn((B, T, N, F))
            y = torch.randn((B, H, N, T_target))
            optimizer.zero_grad()
            pred = model(x, adj)
            loss = loss_fn(pred, y, mask=mask)
            loss.backward()
            optimizer.step()
        t1 = time.time()
        print(f"Threads={threads}: 5 batches took {t1 - t0:.3f}s ({(t1 - t0)/5:.4f}s / batch)")
