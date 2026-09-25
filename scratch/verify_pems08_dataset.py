import numpy as np
import csv
import math
from pathlib import Path

npz_path = Path("data/research/pems08/PEMS08.npz")
csv_path = Path("data/research/pems08/PEMS08.csv")

print("=" * 60)
print("PEMS08 DATASET VERIFICATION AUDIT")
print("=" * 60)

raw = np.load(npz_path)
print("NPZ files/arrays:", list(raw.keys()))
data = raw['data']
print("Data array shape:", data.shape)
print("Data dtype:", data.dtype)
print("Timesteps:", data.shape[0])
print("Nodes:", data.shape[1])
print("Features/Channels:", data.shape[2])

# Check for NaN, Inf, Min, Max per channel
channel_names = ["flow_rate", "occupancy", "speed"]
for i in range(data.shape[2]):
    ch = data[:, :, i]
    nan_count = np.isnan(ch).sum()
    inf_count = np.isinf(ch).sum()
    min_val = np.min(ch)
    max_val = np.max(ch)
    mean_val = np.mean(ch)
    std_val = np.std(ch)
    print(f"Channel {i} ({channel_names[i]}): NaN={nan_count}, Inf={inf_count}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")

# Verify CSV graph topology
print("\n" + "=" * 60)
print("GRAPH TOPOLOGY AUDIT (PEMS08.csv)")
print("=" * 60)

num_nodes = data.shape[1]
adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
edge_count = 0
distances = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    print("CSV columns:", reader.fieldnames)
    for row in reader:
        u = int(row['from'])
        v = int(row['to'])
        dist = float(row['cost'])
        edge_count += 1
        distances.append(dist)
        adj[u, v] = math.exp(-((dist / 1000.0) ** 2))

print(f"Total directed rows in CSV: {edge_count}")
print(f"Distances (cost): min={min(distances):.2f}, max={max(distances):.2f}, mean={np.mean(distances):.2f}")

# Adjacency properties
nonzero_directed = np.count_nonzero(adj)
# Symmetrized adjacency (as used in RealTrafficDatasetAdapter)
adj_sym = adj.copy()
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj[i, j] > 0 and adj[j, i] == 0:
            adj_sym[j, i] = adj[i, j]

nonzero_sym = np.count_nonzero(adj_sym)
density_directed = nonzero_directed / (num_nodes * num_nodes)
density_sym = nonzero_sym / (num_nodes * num_nodes)
sparsity_directed = 1.0 - density_directed
sparsity_sym = 1.0 - density_sym

# Connected components
visited = set()
def dfs(node, visited_set):
    visited_set.add(node)
    for neighbor in range(num_nodes):
        if (adj_sym[node, neighbor] > 0 or adj_sym[neighbor, node] > 0) and neighbor not in visited_set:
            dfs(neighbor, visited_set)

components = 0
isolated_nodes = []
for node in range(num_nodes):
    neighbors = np.where((adj_sym[node, :] > 0) | (adj_sym[:, node] > 0))[0]
    if len(neighbors) == 0:
        isolated_nodes.append(node)
    if node not in visited:
        components += 1
        dfs(node, visited)

print(f"Directed Non-zero entries: {nonzero_directed}")
print(f"Symmetrized Non-zero entries: {nonzero_sym}")
print(f"Directed Sparsity: {sparsity_directed:.4%}")
print(f"Symmetrized Sparsity: {sparsity_sym:.4%}")
print(f"Self-loops in raw CSV: {np.diag(adj).sum()}")
print(f"Connected components (undirected graph): {components}")
print(f"Isolated nodes count: {len(isolated_nodes)}")
if isolated_nodes:
    print(f"Isolated node indices: {isolated_nodes}")
