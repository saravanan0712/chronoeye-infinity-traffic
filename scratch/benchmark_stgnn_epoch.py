import time
import torch
import sys
from pathlib import Path

repo_root = Path(".").resolve()
sys.path.insert(0, str(repo_root / "backend"))

from app.models.stgnn.config import STGNNConfig
from app.models.stgnn.model import SpatioTemporalGNN
from app.models.stgnn.loss import MaskedLoss

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

model = SpatioTemporalGNN(config)
loss_fn = MaskedLoss(loss_type="smooth_l1")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

adj = torch.rand((170, 170))

# Simulate 10 batches of batch_size=32
B = 32
T = 12
N = 170
F = 8
H = 3
T_target = 4

print(f"Model parameters: {model.get_parameter_count()}")

t0 = time.time()
for i in range(10):
    x = torch.randn((B, T, N, F))
    y = torch.randn((B, H, N, T_target))
    mask = torch.ones((B, H, N, T_target), dtype=torch.bool)

    optimizer.zero_grad()
    pred = model(x, adj)
    loss = loss_fn(pred, y, mask=mask)
    loss.backward()
    optimizer.step()

t1 = time.time()
time_per_10_batches = t1 - t0
print(f"10 batches took {time_per_10_batches:.3f}s ({time_per_10_batches/10:.4f}s per batch)")
# Total training batches for 12,489 samples with batch_size=32 = 391 batches
est_epoch_time = (time_per_10_batches / 10) * 391
print(f"Estimated time per epoch (391 batches): {est_epoch_time:.2f}s")
