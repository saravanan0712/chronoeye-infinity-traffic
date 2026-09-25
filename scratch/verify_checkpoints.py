import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from app.forecasting.forecasting_engine import TrafficForecastingEngine
from app.models.stgnn.model import SpatioTemporalGNN

def test_checkpoint(checkpoint_path):
    print(f"\n==========================================")
    print(f"Testing Checkpoint: {checkpoint_path}")
    print(f"==========================================")
    
    engine = TrafficForecastingEngine()
    loaded = engine.load_canonical_stgnn(checkpoint_path)
    print(f"Loaded successfully: {loaded}")
    
    model = engine.get_canonical_stgnn()
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Check num_nodes from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Get num_nodes from checkpoint output heads or config
    num_nodes = ckpt.get("config", {}).get("num_nodes", 10)
    history_window = ckpt.get("config", {}).get("history_window", 12)
    in_channels = ckpt.get("config", {}).get("in_channels", 8)
    
    print(f"Config num_nodes: {num_nodes}, history_window: {history_window}, in_channels: {in_channels}")
    
    # Build dummy input
    # Shape: (B, T_in, N, F_in)
    torch.manual_seed(42)
    x = torch.randn(1, history_window, num_nodes, in_channels)
    adj = torch.eye(num_nodes)
    
    with torch.no_grad():
        out = model(x, adj)
        
    print(f"Input Shape: {list(x.shape)}")
    print(f"Output Shape: {list(out.shape)}")
    
    # Output shape is [B, H, N, 4] where H=3 horizons (+5m, +10m, +15m)
    # Features: [flow, speed, density, occupancy]
    h5_pred = out[0, 0, 0, :].tolist()  # Node 0, Horizon 0 (+5 min)
    h10_pred = out[0, 1, 0, :].tolist() # Node 0, Horizon 1 (+10 min)
    h15_pred = out[0, 2, 0, :].tolist() # Node 0, Horizon 2 (+15 min)
    
    print(f"\nSample Predictions for Node 0 (flow, speed, density, occupancy):")
    print(f"  +5  min horizon: {[round(v, 4) for v in h5_pred]}")
    print(f"  +10 min horizon: {[round(v, 4) for v in h10_pred]}")
    print(f"  +15 min horizon: {[round(v, 4) for v in h15_pred]}")
    
    # Multi-node mean predictions across all nodes
    print(f"\nNetwork-wide Mean Predictions across all {num_nodes} nodes:")
    print(f"  +5  min: Flow={out[0, 0, :, 0].mean().item():.4f}, Speed={out[0, 0, :, 1].mean().item():.4f}, Density={out[0, 0, :, 2].mean().item():.4f}, Occ={out[0, 0, :, 3].mean().item():.4f}")
    print(f"  +10 min: Flow={out[0, 1, :, 0].mean().item():.4f}, Speed={out[0, 1, :, 1].mean().item():.4f}, Density={out[0, 1, :, 2].mean().item():.4f}, Occ={out[0, 1, :, 3].mean().item():.4f}")
    print(f"  +15 min: Flow={out[0, 2, :, 0].mean().item():.4f}, Speed={out[0, 2, :, 1].mean().item():.4f}, Density={out[0, 2, :, 2].mean().item():.4f}, Occ={out[0, 2, :, 3].mean().item():.4f}")

if __name__ == "__main__":
    test_checkpoint("experiments/module3/research_exp_01/checkpoints/best_stgnn_model.pt")
    test_checkpoint("experiments/module3/research_exp_02/checkpoints/best_stgnn_model.pt")
