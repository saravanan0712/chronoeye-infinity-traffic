"""
ChronoEye Infinity - Module 3 Step 4: Spatio-Temporal Dataset Sequence Slicer & Normalization Pipeline.
Converts discrete TemporalGraphDatasetContract snapshots G(t0), G(t1), ... into supervised
spatio-temporal sequences (X, Y) for ST-GNN models with strict chronological train/val/test splitting,
TRAIN-only feature/target normalization, deterministic node ordering, and leakage prevention.
"""

import math
import uuid
import statistics
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from typing import List, Dict, Optional, Any, Union, Tuple, Sequence
from pydantic import BaseModel, Field

from app.graph.graph_schema import NodeType, EdgeType
from app.forecasting.forecasting_schema import ForecastHorizon
from app.graph.temporal_snapshot_schema import (
    TemporalNodeFeatures,
    TemporalEdgeFeatures,
    TemporalGraphSnapshot,
    TemporalGraphDatasetContract,
)


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class NormalizationParams(BaseModel):
    """
    Normalization parameters fitted strictly on the TRAINING split.
    """
    feature_names: List[str]
    mean: Dict[str, float] = Field(default_factory=dict)
    std: Dict[str, float] = Field(default_factory=dict)
    min_val: Dict[str, float] = Field(default_factory=dict)
    max_val: Dict[str, float] = Field(default_factory=dict)
    method: str = "zscore"  # "zscore", "minmax", "none"


class AdjacencyData(BaseModel):
    """
    Deterministic graph adjacency representation for ST-GNN models.
    """
    node_ids: List[str]
    node_to_idx: Dict[str, int]
    num_nodes: int
    edge_index: List[List[int]] = Field(default_factory=lambda: [[], []])  # [2, num_edges] (src_idx, dst_idx)
    edge_weights: List[float] = Field(default_factory=list)
    adjacency_matrix: List[List[float]] = Field(default_factory=list)      # [N, N]
    has_unobserved_gaps: Dict[str, bool] = Field(default_factory=dict)     # "src->dst" -> bool


class SpatioTemporalSample(BaseModel):
    """
    Single supervised spatio-temporal training/evaluation sample.
    X: Historical input sequence [T_in, N, F]
    Y: Future multi-horizon targets [T_out, N, T_target]
    """
    sample_id: str = Field(default_factory=lambda: f"SMP_{uuid.uuid4().hex[:8]}")
    split: DatasetSplit
    input_start_time: float
    input_end_time: float
    target_times: Dict[str, float] = Field(default_factory=dict)  # horizon_name -> target_timestamp
    
    # Feature matrices: [T_in, N, F]
    # x values can be None if unobserved; x_mask is True for valid measured values, False for missing
    x_raw: List[List[List[Optional[float]]]] = Field(default_factory=list)
    x_normalized: List[List[List[Optional[float]]]] = Field(default_factory=list)
    x_mask: List[List[List[bool]]] = Field(default_factory=list)
    
    # Target matrices: [T_out, N, T_target]
    y_raw: List[List[List[Optional[float]]]] = Field(default_factory=list)
    y_normalized: List[List[List[Optional[float]]]] = Field(default_factory=list)
    y_mask: List[List[List[bool]]] = Field(default_factory=list)
    
    horizon_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LazySpatioTemporalSplit(Dataset):
    """
    Lazy PyTorch Dataset for mini-batch generation.
    Slices historical inputs [T_in, N, F] and multi-horizon targets [H, N, Target_F]
    on-the-fly from pre-normalized compact tensors with zero memory duplication.
    """
    def __init__(
        self,
        norm_features: torch.Tensor,       # [T, N, F] float32
        norm_targets: torch.Tensor,        # [T, N, Target_F] float32
        raw_features: torch.Tensor,        # [T, N, F] float32
        raw_targets: torch.Tensor,         # [T, N, Target_F] float32
        feature_mask: torch.Tensor,        # [F] bool
        target_mask: torch.Tensor,         # [Target_F] bool
        input_sequence_length: int,
        forecast_horizons: List[Tuple[str, int]],
        sample_start_indices: List[int],
        split: DatasetSplit,
        feature_names: List[str],
        target_names: List[str],
        node_ids: List[str],
        start_time_offset: float = 0.0,
        window_seconds: float = 300.0,
    ):
        self.norm_features = norm_features
        self.norm_targets = norm_targets
        self.raw_features = raw_features
        self.raw_targets = raw_targets
        self.feature_mask = feature_mask
        self.target_mask = target_mask
        self.input_sequence_length = input_sequence_length
        self.forecast_horizons = forecast_horizons
        self.sample_start_indices = sample_start_indices
        self.split = split
        self.feature_names = feature_names
        self.target_names = target_names
        self.node_ids = node_ids
        self.num_nodes = len(node_ids)
        self.start_time_offset = start_time_offset
        self.window_seconds = window_seconds

        # Pre-expand static boolean masks for O(1) mini-batch collation
        self._x_mask = self.feature_mask.view(1, 1, -1).expand(
            self.input_sequence_length, self.num_nodes, -1
        ).clone()
        self._y_mask = self.target_mask.view(1, 1, -1).expand(
            len(self.forecast_horizons), self.num_nodes, -1
        ).clone()

    def __len__(self) -> int:
        return len(self.sample_start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.sample_start_indices[idx]
        x = self.norm_features[t : t + self.input_sequence_length]  # [T_in, N, F]
        x_mask = self._x_mask                                        # [T_in, N, F]

        y_steps = [t + self.input_sequence_length - 1 + h_step for _, h_step in self.forecast_horizons]
        y = self.norm_targets[y_steps]                               # [H, N, Target_F]
        y_mask = self._y_mask                                        # [H, N, Target_F]
        return x, x_mask, y, y_mask

    def get_sample(self, idx: int) -> SpatioTemporalSample:
        """Constructs an individual SpatioTemporalSample on demand for legacy inspection."""
        t = self.sample_start_indices[idx]
        t_in = self.input_sequence_length
        input_start = self.start_time_offset + float(t * self.window_seconds)
        input_end = self.start_time_offset + float((t + t_in) * self.window_seconds)

        target_times = {}
        horizon_names = []
        for h_name, h_step in self.forecast_horizons:
            target_t = t + t_in - 1 + h_step
            target_times[h_name] = self.start_time_offset + float(target_t * self.window_seconds)
            horizon_names.append(h_name)

        x_raw_t = self.raw_features[t : t + t_in].tolist()
        x_norm_t = self.norm_features[t : t + t_in].tolist()
        x_mask_t = self._x_mask.tolist()

        y_steps = [t + t_in - 1 + h_step for _, h_step in self.forecast_horizons]
        y_raw_t = self.raw_targets[y_steps].tolist()
        y_norm_t = self.norm_targets[y_steps].tolist()
        y_mask_t = self._y_mask.tolist()

        # Set unobserved entries where mask is False to None in raw/normalized lists
        f_mask_np = self.feature_mask.numpy()
        for step in range(len(x_raw_t)):
            for node in range(len(x_raw_t[step])):
                for f in range(len(f_mask_np)):
                    if not f_mask_np[f]:
                        x_raw_t[step][node][f] = None
                        x_norm_t[step][node][f] = None

        t_mask_np = self.target_mask.numpy()
        for h in range(len(y_raw_t)):
            for node in range(len(y_raw_t[h])):
                for tf in range(len(t_mask_np)):
                    if not t_mask_np[tf]:
                        y_raw_t[h][node][tf] = None
                        y_norm_t[h][node][tf] = None

        return SpatioTemporalSample(
            sample_id=f"SMP_{self.split.value}_{int(input_start)}_{int(input_end)}",
            split=self.split,
            input_start_time=input_start,
            input_end_time=input_end,
            target_times=target_times,
            horizon_names=horizon_names,
            x_raw=x_raw_t,
            x_normalized=x_norm_t,
            x_mask=x_mask_t,
            y_raw=y_raw_t,
            y_normalized=y_norm_t,
            y_mask=y_mask_t,
        )


class LazySampleListProxy(Sequence):
    """
    List-like proxy over LazySpatioTemporalSplit allowing len(), indexing, slicing,
    and iteration without eagerly instantiating thousands of SpatioTemporalSample objects.
    """
    def __init__(self, split_dataset: LazySpatioTemporalSplit):
        self._split = split_dataset

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            return [self._split.get_sample(i) for i in range(*idx.indices(len(self._split)))]
        if idx < 0:
            idx += len(self._split)
        if idx < 0 or idx >= len(self._split):
            raise IndexError(f"Sample index {idx} out of range (length {len(self._split)})")
        return self._split.get_sample(idx)

    def __iter__(self):
        for i in range(len(self._split)):
            yield self._split.get_sample(i)

    def __bool__(self) -> bool:
        return len(self._split) > 0

    @property
    def split_dataset(self) -> LazySpatioTemporalSplit:
        return self._split


class CompactSpatioTemporalDataset:
    """
    Memory-hardened spatio-temporal dataset container for ST-GNN training.
    Provides PyTorch DataLoaders with on-demand mini-batch collation and train-only normalization.
    """
    def __init__(
        self,
        dataset_id: str,
        input_sequence_length: int,
        forecast_horizons: List[str],
        feature_names: List[str],
        target_names: List[str],
        adjacency: AdjacencyData,
        feature_scaler: NormalizationParams,
        target_scaler: NormalizationParams,
        train_dataset: LazySpatioTemporalSplit,
        val_dataset: LazySpatioTemporalSplit,
        test_dataset: LazySpatioTemporalSplit,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_id = dataset_id
        self.input_sequence_length = input_sequence_length
        self.forecast_horizons = forecast_horizons
        self.feature_names = feature_names
        self.target_names = target_names
        self.adjacency = adjacency
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_samples = LazySampleListProxy(train_dataset)
        self.val_samples = LazySampleListProxy(val_dataset)
        self.test_samples = LazySampleListProxy(test_dataset)
        self.metadata = metadata or {}

    @property
    def total_samples(self) -> int:
        return len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)

    @property
    def num_nodes(self) -> int:
        return self.adjacency.num_nodes

    def get_dataloader(
        self,
        split: Union[str, DatasetSplit],
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """Constructs a PyTorch DataLoader for the requested split."""
        split_name = split.value if isinstance(split, DatasetSplit) else str(split).lower()
        if split_name in ("train", "training"):
            ds = self.train_dataset
        elif split_name in ("val", "validation", "valid"):
            ds = self.val_dataset
        elif split_name in ("test", "testing"):
            ds = self.test_dataset
        else:
            raise ValueError(f"Unknown dataset split '{split}'. Expected 'train', 'val', or 'test'.")

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )


class SpatioTemporalDataset(BaseModel):
    """
    Full structured spatio-temporal dataset with chronological splits,
    fitted normalization parameters, graph adjacency, and metadata.
    """
    dataset_id: str = Field(default_factory=lambda: f"ST_DS_{uuid.uuid4().hex[:8]}")
    input_sequence_length: int
    forecast_horizons: List[str]
    feature_names: List[str]
    target_names: List[str]
    
    adjacency: AdjacencyData
    feature_scaler: NormalizationParams
    target_scaler: NormalizationParams
    
    train_samples: List[SpatioTemporalSample] = Field(default_factory=list)
    val_samples: List[SpatioTemporalSample] = Field(default_factory=list)
    test_samples: List[SpatioTemporalSample] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        return len(self.train_samples) + len(self.val_samples) + len(self.test_samples)

    @property
    def num_nodes(self) -> int:
        return self.adjacency.num_nodes



class TemporalDatasetSlicer:
    """
    Spatio-Temporal Sequence Slicer & Normalization Engine.
    Converts temporal graph snapshots into leakage-free supervised ST-GNN sequences.
    """

    DEFAULT_FEATURES = [
        "vehicle_count",
        "flow_rate",
        "density",
        "average_speed",
        "queue_length",
        "congestion",
        "incoming_flow",
        "outgoing_flow",
    ]

    DEFAULT_TARGETS = [
        "flow_rate",
        "density",
        "congestion",
        "travel_time",
    ]

    def __init__(
        self,
        input_sequence_length: int = 3,
        forecast_horizons: Optional[List[Union[ForecastHorizon, str, int]]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        normalization_method: str = "zscore",
        max_allowed_gap_seconds: Optional[float] = None,
    ):
        """
        Initializes sequence slicer with configurable history length, forecast horizons, and split ratios.
        """
        self.input_sequence_length = int(input_sequence_length)
        self.forecast_horizons = self._normalize_horizons(forecast_horizons)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.feature_names = feature_names or self.DEFAULT_FEATURES
        self.target_names = target_names or self.DEFAULT_TARGETS
        self.normalization_method = normalization_method
        self.max_allowed_gap_seconds = max_allowed_gap_seconds

    def _normalize_horizons(
        self, horizons: Optional[List[Union[ForecastHorizon, str, int]]]
    ) -> List[Tuple[str, int]]:
        """
        Normalizes horizons into (name, step_offset) tuples.
        For 5-min snapshots:
        PLUS_5MIN / 5 -> 1 step
        PLUS_10MIN / 10 -> 2 steps
        PLUS_15MIN / 15 -> 3 steps
        PLUS_30MIN / 30 -> 6 steps
        """
        if not horizons:
            return [
                ("PLUS_5MIN", 1),
                ("PLUS_10MIN", 2),
                ("PLUS_15MIN", 3),
            ]

        res = []
        for h in horizons:
            if isinstance(h, ForecastHorizon):
                name = h.value
            elif isinstance(h, str):
                name = h
            elif isinstance(h, int):
                name = f"PLUS_{h * 5}MIN" if h > 0 else "PLUS_0MIN"
            else:
                name = str(h)

            # Map to step offset
            if name in ("PLUS_5MIN", "5", "5MIN", "5_MIN"):
                res.append(("PLUS_5MIN", 1))
            elif name in ("PLUS_10MIN", "10", "10MIN", "10_MIN"):
                res.append(("PLUS_10MIN", 2))
            elif name in ("PLUS_15MIN", "15", "15MIN", "15_MIN"):
                res.append(("PLUS_15MIN", 3))
            elif name in ("PLUS_30MIN", "30", "30MIN", "30_MIN"):
                res.append(("PLUS_30MIN", 6))
            elif isinstance(h, int):
                res.append((name, h))
            else:
                res.append((name, 1))

        # Sort by step offset for deterministic ordering
        res.sort(key=lambda x: x[1])
        return res

    def slice_dataset(
        self,
        contract: Union[TemporalGraphDatasetContract, List[TemporalGraphSnapshot], Any],
    ) -> Union[SpatioTemporalDataset, CompactSpatioTemporalDataset]:
        """
        Main pipeline: Slices snapshots or compact benchmark into supervised sequences,
        splits chronologically, and normalizes features/targets using TRAIN statistics only.
        """
        if hasattr(contract, "features") and hasattr(contract, "adjacency_matrix"):
            return self.slice_compact_dataset(contract)

        snapshots = contract.snapshots if isinstance(contract, TemporalGraphDatasetContract) else contract
        
        # 1. Sort snapshots chronologically
        sorted_snaps = sorted(snapshots, key=lambda s: s.start_time)

        
        if not sorted_snaps:
            empty_adj = AdjacencyData(node_ids=[], node_to_idx={}, num_nodes=0)
            empty_norm = NormalizationParams(feature_names=self.feature_names, method=self.normalization_method)
            empty_tgt_norm = NormalizationParams(feature_names=self.target_names, method=self.normalization_method)
            return SpatioTemporalDataset(
                input_sequence_length=self.input_sequence_length,
                forecast_horizons=[h[0] for h in self.forecast_horizons],
                feature_names=self.feature_names,
                target_names=self.target_names,
                adjacency=empty_adj,
                feature_scaler=empty_norm,
                target_scaler=empty_tgt_norm,
                train_samples=[],
                val_samples=[],
                test_samples=[],
            )

        # 2. Build deterministic node mapping and adjacency from data
        node_ids, node_to_idx = self._extract_node_mapping(sorted_snaps)
        adjacency = self._build_adjacency(sorted_snaps, node_ids, node_to_idx)

        # 3. Detect temporal continuous segments (gaps check)
        stride = sorted_snaps[0].duration_seconds or 300.0
        max_gap = self.max_allowed_gap_seconds if self.max_allowed_gap_seconds is not None else (stride * 1.5)
        
        segments: List[List[TemporalGraphSnapshot]] = []
        curr_segment: List[TemporalGraphSnapshot] = [sorted_snaps[0]]
        
        for idx in range(1, len(sorted_snaps)):
            prev_s = sorted_snaps[idx - 1]
            curr_s = sorted_snaps[idx]
            dt = curr_s.start_time - prev_s.start_time
            if dt > max_gap:
                # Gap detected -> start new continuous segment
                segments.append(curr_segment)
                curr_segment = [curr_s]
            else:
                curr_segment.append(curr_s)
        if curr_segment:
            segments.append(curr_segment)

        # 4. Generate all valid continuous raw samples from continuous segments
        all_raw_samples: List[Dict[str, Any]] = []
        for seg in segments:
            all_raw_samples.extend(self._extract_samples_from_segment(seg, 0, len(seg), node_ids))

        # Sort strictly by input_start_time
        all_raw_samples.sort(key=lambda s: s["input_start_time"])
        total_n = len(all_raw_samples)

        if total_n == 0:
            train_raw, val_raw, test_raw = [], [], []
        elif total_n == 1:
            train_raw = all_raw_samples if self.train_ratio > 0 else []
            val_raw = all_raw_samples if (not train_raw and self.val_ratio > 0) else []
            test_raw = all_raw_samples if (not train_raw and not val_raw) else []
        elif self.train_ratio >= 0.999:
            train_raw, val_raw, test_raw = all_raw_samples, [], []
        elif self.val_ratio >= 0.999:
            train_raw, val_raw, test_raw = [], all_raw_samples, []
        elif self.test_ratio >= 0.999:
            train_raw, val_raw, test_raw = [], [], all_raw_samples
        else:
            train_count = int(total_n * self.train_ratio)
            val_count = int(total_n * self.val_ratio)
            
            # Ensure at least 1 sample in train/val/test if ratios are positive
            if self.train_ratio > 0 and train_count == 0 and total_n > 0:
                train_count = 1
            if self.val_ratio > 0 and val_count == 0 and (total_n - train_count) > 1:
                val_count = 1

            test_count = max(0, total_n - train_count - val_count)

            train_raw = all_raw_samples[:train_count]
            val_raw = all_raw_samples[train_count : train_count + val_count]
            test_raw = all_raw_samples[train_count + val_count :]

        # 5. Fit Normalization Scalers on TRAIN ONLY
        feature_scaler = self._fit_feature_scaler(train_raw, self.feature_names)
        target_scaler = self._fit_target_scaler(train_raw, self.target_names)

        # 6. Build SpatioTemporalSample objects with normalized arrays
        train_samples = self._build_samples(train_raw, DatasetSplit.TRAIN, feature_scaler, target_scaler)
        val_samples = self._build_samples(val_raw, DatasetSplit.VAL, feature_scaler, target_scaler)
        test_samples = self._build_samples(test_raw, DatasetSplit.TEST, feature_scaler, target_scaler)

        return SpatioTemporalDataset(
            dataset_id=f"ST_DS_{int(sorted_snaps[0].start_time)}_{int(sorted_snaps[-1].end_time)}",
            input_sequence_length=self.input_sequence_length,
            forecast_horizons=[h[0] for h in self.forecast_horizons],
            feature_names=self.feature_names,
            target_names=self.target_names,
            adjacency=adjacency,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            metadata={
                "total_snapshots": len(sorted_snaps),
                "continuous_segments_count": len(segments),
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "normalization_method": self.normalization_method,
            },
        )

    def _extract_samples_from_segment(
        self,
        segment: List[TemporalGraphSnapshot],
        start_idx: int,
        end_idx: int,
        node_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Extracts sliding window samples from a contiguous slice of snapshots [start_idx, end_idx).
        """
        sub_seg = segment[start_idx:end_idx]
        seg_len = len(sub_seg)
        max_horizon_step = max(h[1] for h in self.forecast_horizons) if self.forecast_horizons else 1
        required_len = self.input_sequence_length + max_horizon_step

        samples = []
        if seg_len < required_len:
            return samples

        for i in range(seg_len - required_len + 1):
            input_snaps = sub_seg[i : i + self.input_sequence_length]
            input_start = input_snaps[0].start_time
            input_end = input_snaps[-1].end_time

            x_raw, x_mask = self._extract_feature_tensor(input_snaps, node_ids, self.feature_names)

            y_raw: List[List[List[Optional[float]]]] = []
            y_mask: List[List[List[bool]]] = []
            target_times: Dict[str, float] = {}
            horizon_names: List[str] = []

            for h_name, h_step in self.forecast_horizons:
                target_idx = i + self.input_sequence_length - 1 + h_step
                if target_idx < seg_len:
                    target_snap = sub_seg[target_idx]
                    t_raw, t_mask = self._extract_target_tensor([target_snap], node_ids, self.target_names)
                    y_raw.append(t_raw[0])
                    y_mask.append(t_mask[0])
                    target_times[h_name] = target_snap.start_time
                    horizon_names.append(h_name)

            if len(y_raw) == len(self.forecast_horizons):
                samples.append({
                    "input_start_time": input_start,
                    "input_end_time": input_end,
                    "target_times": target_times,
                    "horizon_names": horizon_names,
                    "x_raw": x_raw,
                    "x_mask": x_mask,
                    "y_raw": y_raw,
                    "y_mask": y_mask,
                })

        return samples

    def _extract_node_mapping(
        self, snapshots: List[TemporalGraphSnapshot]
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Extracts unique node IDs across all snapshots in deterministic sorted order.
        """
        all_nodes = set()
        for snap in snapshots:
            for n in snap.nodes:
                all_nodes.add(n.node_id)
            for e in snap.edges:
                all_nodes.add(e.source)
                all_nodes.add(e.target)
        sorted_nodes = sorted(list(all_nodes))
        node_to_idx = {nid: idx for idx, nid in enumerate(sorted_nodes)}
        return sorted_nodes, node_to_idx

    def _build_adjacency(
        self,
        snapshots: List[TemporalGraphSnapshot],
        node_ids: List[str],
        node_to_idx: Dict[str, int],
    ) -> AdjacencyData:
        """
        Constructs deterministic static adjacency matrix and PyG edge index
        from observed transitions across all snapshots.
        """
        N = len(node_ids)
        adj_matrix = [[0.0 for _ in range(N)] for _ in range(N)]
        edge_counts: Dict[Tuple[int, int], int] = {}
        has_gaps_dict: Dict[str, bool] = {}

        for snap in snapshots:
            for e in snap.edges:
                if e.source in node_to_idx and e.target in node_to_idx:
                    u = node_to_idx[e.source]
                    v = node_to_idx[e.target]
                    pair = (u, v)
                    edge_counts[pair] = edge_counts.get(pair, 0) + (e.transition_count or 1)
                    
                    gap_key = f"{e.source}->{e.target}"
                    if e.has_unobserved_gap:
                        has_gaps_dict[gap_key] = True
                    elif gap_key not in has_gaps_dict:
                        has_gaps_dict[gap_key] = False

        # Build edge_index and adjacency matrix
        src_indices = []
        dst_indices = []
        weights = []

        for (u, v) in sorted(edge_counts.keys()):
            cnt = float(edge_counts[(u, v)])
            adj_matrix[u][v] = cnt
            src_indices.append(u)
            dst_indices.append(v)
            weights.append(cnt)

        return AdjacencyData(
            node_ids=node_ids,
            node_to_idx=node_to_idx,
            num_nodes=N,
            edge_index=[src_indices, dst_indices],
            edge_weights=weights,
            adjacency_matrix=adj_matrix,
            has_unobserved_gaps=has_gaps_dict,
        )

    def _extract_feature_tensor(
        self,
        snapshots: List[TemporalGraphSnapshot],
        node_ids: List[str],
        feature_names: List[str],
    ) -> Tuple[List[List[List[Optional[float]]]], List[List[List[bool]]]]:
        """
        Extracts 3D raw feature tensor [T, N, F] and corresponding boolean availability mask.
        """
        tensor: List[List[List[Optional[float]]]] = []
        mask: List[List[List[bool]]] = []

        for snap in snapshots:
            snap_nodes_map = {n.node_id: n for n in snap.nodes}
            t_slice: List[List[Optional[float]]] = []
            m_slice: List[List[bool]] = []

            for n_id in node_ids:
                node_feat = snap_nodes_map.get(n_id)
                n_feats: List[Optional[float]] = []
                n_mask: List[bool] = []

                for fname in feature_names:
                    val = None
                    if node_feat is not None:
                        raw_val = getattr(node_feat, fname, None)
                        if raw_val is not None:
                            val = float(raw_val)
                    
                    n_feats.append(val)
                    n_mask.append(val is not None)

                t_slice.append(n_feats)
                m_slice.append(n_mask)

            tensor.append(t_slice)
            mask.append(m_slice)

        return tensor, mask

    def _extract_target_tensor(
        self,
        snapshots: List[TemporalGraphSnapshot],
        node_ids: List[str],
        target_names: List[str],
    ) -> Tuple[List[List[List[Optional[float]]]], List[List[List[bool]]]]:
        """
        Extracts 3D raw target tensor [T, N, Target_F] and corresponding boolean availability mask.
        """
        return self._extract_feature_tensor(snapshots, node_ids, target_names)

    def _fit_feature_scaler(
        self,
        train_samples: List[Dict[str, Any]],
        feature_names: List[str],
    ) -> NormalizationParams:
        """
        Calculates mean and standard deviation for each feature strictly from TRAIN samples.
        Zero standard deviation is handled safely without division by zero.
        """
        mean_dict = {}
        std_dict = {}
        min_dict = {}
        max_dict = {}

        for f_idx, fname in enumerate(feature_names):
            valid_vals = []
            for s in train_samples:
                x_raw = s["x_raw"]  # [T, N, F]
                for t in range(len(x_raw)):
                    for n in range(len(x_raw[t])):
                        val = x_raw[t][n][f_idx]
                        if val is not None and not math.isnan(val):
                            valid_vals.append(val)

            if valid_vals:
                m = float(statistics.mean(valid_vals))
                s = float(statistics.stdev(valid_vals)) if len(valid_vals) > 1 else 0.0
                mean_dict[fname] = round(m, 4)
                # Handle zero std safely: fallback to 1.0 to prevent division by zero
                std_dict[fname] = round(s, 4) if s > 1e-6 else 1.0
                min_dict[fname] = round(min(valid_vals), 4)
                max_dict[fname] = round(max(valid_vals), 4)
            else:
                mean_dict[fname] = 0.0
                std_dict[fname] = 1.0
                min_dict[fname] = 0.0
                max_dict[fname] = 1.0

        return NormalizationParams(
            feature_names=feature_names,
            mean=mean_dict,
            std=std_dict,
            min_val=min_dict,
            max_val=max_dict,
            method=self.normalization_method,
        )

    def _fit_target_scaler(
        self,
        train_samples: List[Dict[str, Any]],
        target_names: List[str],
    ) -> NormalizationParams:
        """
        Calculates mean and standard deviation for target metrics strictly from TRAIN samples.
        """
        mean_dict = {}
        std_dict = {}
        min_dict = {}
        max_dict = {}

        for t_idx, tname in enumerate(target_names):
            valid_vals = []
            for s in train_samples:
                y_raw = s["y_raw"]  # [T_out, N, Target_F]
                for t in range(len(y_raw)):
                    for n in range(len(y_raw[t])):
                        val = y_raw[t][n][t_idx]
                        if val is not None and not math.isnan(val):
                            valid_vals.append(val)

            if valid_vals:
                m = float(statistics.mean(valid_vals))
                s = float(statistics.stdev(valid_vals)) if len(valid_vals) > 1 else 0.0
                mean_dict[tname] = round(m, 4)
                std_dict[tname] = round(s, 4) if s > 1e-6 else 1.0
                min_dict[tname] = round(min(valid_vals), 4)
                max_dict[tname] = round(max(valid_vals), 4)
            else:
                mean_dict[tname] = 0.0
                std_dict[tname] = 1.0
                min_dict[tname] = 0.0
                max_dict[tname] = 1.0

        return NormalizationParams(
            feature_names=target_names,
            mean=mean_dict,
            std=std_dict,
            min_val=min_dict,
            max_val=max_dict,
            method=self.normalization_method,
        )

    def _normalize_tensor(
        self,
        tensor: List[List[List[Optional[float]]]],
        scaler: NormalizationParams,
        field_names: List[str],
    ) -> List[List[List[Optional[float]]]]:
        """
        Transforms 3D raw tensor using precomputed scaler parameters.
        Missing values (None) remain None.
        """
        norm_tensor: List[List[List[Optional[float]]]] = []

        for t in range(len(tensor)):
            t_slice: List[List[Optional[float]]] = []
            for n in range(len(tensor[t])):
                n_slice: List[Optional[float]] = []
                for f_idx, fname in enumerate(field_names):
                    val = tensor[t][n][f_idx]
                    if val is None:
                        n_slice.append(None)
                    else:
                        m = scaler.mean.get(fname, 0.0)
                        s = scaler.std.get(fname, 1.0)
                        norm_val = (val - m) / (s if s > 1e-6 else 1.0)
                        n_slice.append(round(norm_val, 4))
                t_slice.append(n_slice)
            norm_tensor.append(t_slice)

        return norm_tensor

    def _build_samples(
        self,
        raw_samples: List[Dict[str, Any]],
        split: DatasetSplit,
        feature_scaler: NormalizationParams,
        target_scaler: NormalizationParams,
    ) -> List[SpatioTemporalSample]:
        """
        Constructs SpatioTemporalSample models with both raw and normalized arrays.
        """
        samples = []
        for s in raw_samples:
            x_norm = self._normalize_tensor(s["x_raw"], feature_scaler, self.feature_names)
            y_norm = self._normalize_tensor(s["y_raw"], target_scaler, self.target_names)

            samples.append(
                SpatioTemporalSample(
                    sample_id=f"SMP_{split.value}_{int(s['input_start_time'])}_{int(s['input_end_time'])}",
                    split=split,
                    input_start_time=s["input_start_time"],
                    input_end_time=s["input_end_time"],
                    target_times=s["target_times"],
                    horizon_names=s["horizon_names"],
                    x_raw=s["x_raw"],
                    x_normalized=x_norm,
                    x_mask=s["x_mask"],
                    y_raw=s["y_raw"],
                    y_normalized=y_norm,
                    y_mask=s["y_mask"],
                )
            )
        return samples

    def slice_compact_dataset(
        self,
        compact_data: Any,
    ) -> CompactSpatioTemporalDataset:
        """
        Slices compact benchmark dataset into train/val/test splits without duplicating
        overlapping sliding windows. Fits normalization scalers strictly on TRAIN samples.
        """
        features_np = compact_data.features.astype(np.float32)  # [T, N, F]
        targets_np = compact_data.targets.astype(np.float32)    # [T, N, Target_F]

        T_total = compact_data.num_timesteps
        N_nodes = compact_data.num_nodes

        max_h = max(h[1] for h in self.forecast_horizons) if self.forecast_horizons else 1
        required_len = self.input_sequence_length + max_h

        total_samples = max(0, T_total - required_len + 1)

        if total_samples == 0:
            train_count, val_count, test_count = 0, 0, 0
        elif total_samples == 1:
            train_count = 1 if self.train_ratio > 0 else 0
            val_count = 1 if (train_count == 0 and self.val_ratio > 0) else 0
            test_count = 1 if (train_count == 0 and val_count == 0) else 0
        elif self.train_ratio >= 0.999:
            train_count, val_count, test_count = total_samples, 0, 0
        elif self.val_ratio >= 0.999:
            train_count, val_count, test_count = 0, total_samples, 0
        elif self.test_ratio >= 0.999:
            train_count, val_count, test_count = 0, 0, total_samples
        else:
            train_count = int(total_samples * self.train_ratio)
            val_count = int(total_samples * self.val_ratio)
            if self.train_ratio > 0 and train_count == 0 and total_samples > 0:
                train_count = 1
            if self.val_ratio > 0 and val_count == 0 and (total_samples - train_count) > 1:
                val_count = 1
            test_count = max(0, total_samples - train_count - val_count)

        train_indices = list(range(0, train_count))
        val_indices = list(range(train_count, train_count + val_count))
        test_indices = list(range(train_count + val_count, total_samples))

        # Fit Scalers strictly on TRAIN split samples
        feature_names = compact_data.feature_names
        target_names = compact_data.target_names

        f_mean, f_std, f_min, f_max = {}, {}, {}, {}
        for f_idx, fname in enumerate(feature_names):
            if compact_data.feature_mask[f_idx] and train_count > 0:
                # Collect valid observations across all training input windows
                train_vals = np.concatenate([
                    features_np[i : i + self.input_sequence_length, :, f_idx].ravel()
                    for i in train_indices
                ])
                m = float(np.mean(train_vals))
                s = float(np.std(train_vals, ddof=1)) if len(train_vals) > 1 else 0.0
                f_mean[fname] = round(m, 4)
                f_std[fname] = round(s, 4) if s > 1e-6 else 1.0
                f_min[fname] = round(float(np.min(train_vals)), 4)
                f_max[fname] = round(float(np.max(train_vals)), 4)
            else:
                f_mean[fname] = 0.0
                f_std[fname] = 1.0
                f_min[fname] = 0.0
                f_max[fname] = 1.0

        feature_scaler = NormalizationParams(
            feature_names=feature_names,
            mean=f_mean,
            std=f_std,
            min_val=f_min,
            max_val=f_max,
            method=self.normalization_method,
        )

        t_mean, t_std, t_min, t_max = {}, {}, {}, {}
        for t_idx, tname in enumerate(target_names):
            if compact_data.target_mask[t_idx] and train_count > 0:
                # Collect valid target observations across all training horizons
                train_tgt_vals = np.concatenate([
                    np.concatenate([
                        targets_np[i + self.input_sequence_length - 1 + h_step, :, t_idx].ravel()
                        for _, h_step in self.forecast_horizons
                    ])
                    for i in train_indices
                ])
                m = float(np.mean(train_tgt_vals))
                s = float(np.std(train_tgt_vals, ddof=1)) if len(train_tgt_vals) > 1 else 0.0
                t_mean[tname] = round(m, 4)
                t_std[tname] = round(s, 4) if s > 1e-6 else 1.0
                t_min[tname] = round(float(np.min(train_tgt_vals)), 4)
                t_max[tname] = round(float(np.max(train_tgt_vals)), 4)
            else:
                t_mean[tname] = 0.0
                t_std[tname] = 1.0
                t_min[tname] = 0.0
                t_max[tname] = 1.0

        target_scaler = NormalizationParams(
            feature_names=target_names,
            mean=t_mean,
            std=t_std,
            min_val=t_min,
            max_val=t_max,
            method=self.normalization_method,
        )

        # Pre-normalize compact tensors in float32
        norm_features_np = np.zeros_like(features_np, dtype=np.float32)
        for f_idx, fname in enumerate(feature_names):
            if compact_data.feature_mask[f_idx]:
                m = feature_scaler.mean.get(fname, 0.0)
                s = feature_scaler.std.get(fname, 1.0)
                norm_features_np[:, :, f_idx] = (features_np[:, :, f_idx] - m) / (s if s > 1e-6 else 1.0)

        norm_targets_np = np.zeros_like(targets_np, dtype=np.float32)
        for t_idx, tname in enumerate(target_names):
            if compact_data.target_mask[t_idx]:
                m = target_scaler.mean.get(tname, 0.0)
                s = target_scaler.std.get(tname, 1.0)
                norm_targets_np[:, :, t_idx] = (targets_np[:, :, t_idx] - m) / (s if s > 1e-6 else 1.0)

        # Convert to PyTorch Tensors
        norm_features_tensor = torch.from_numpy(norm_features_np)
        norm_targets_tensor = torch.from_numpy(norm_targets_np)
        raw_features_tensor = torch.from_numpy(features_np)
        raw_targets_tensor = torch.from_numpy(targets_np)
        feature_mask_tensor = torch.from_numpy(compact_data.feature_mask.astype(bool))
        target_mask_tensor = torch.from_numpy(compact_data.target_mask.astype(bool))

        train_split = LazySpatioTemporalSplit(
            norm_features=norm_features_tensor,
            norm_targets=norm_targets_tensor,
            raw_features=raw_features_tensor,
            raw_targets=raw_targets_tensor,
            feature_mask=feature_mask_tensor,
            target_mask=target_mask_tensor,
            input_sequence_length=self.input_sequence_length,
            forecast_horizons=self.forecast_horizons,
            sample_start_indices=train_indices,
            split=DatasetSplit.TRAIN,
            feature_names=feature_names,
            target_names=target_names,
            node_ids=compact_data.node_ids,
            start_time_offset=0.0,
            window_seconds=compact_data.window_seconds,
        )

        val_split = LazySpatioTemporalSplit(
            norm_features=norm_features_tensor,
            norm_targets=norm_targets_tensor,
            raw_features=raw_features_tensor,
            raw_targets=raw_targets_tensor,
            feature_mask=feature_mask_tensor,
            target_mask=target_mask_tensor,
            input_sequence_length=self.input_sequence_length,
            forecast_horizons=self.forecast_horizons,
            sample_start_indices=val_indices,
            split=DatasetSplit.VAL,
            feature_names=feature_names,
            target_names=target_names,
            node_ids=compact_data.node_ids,
            start_time_offset=0.0,
            window_seconds=compact_data.window_seconds,
        )

        test_split = LazySpatioTemporalSplit(
            norm_features=norm_features_tensor,
            norm_targets=norm_targets_tensor,
            raw_features=raw_features_tensor,
            raw_targets=raw_targets_tensor,
            feature_mask=feature_mask_tensor,
            target_mask=target_mask_tensor,
            input_sequence_length=self.input_sequence_length,
            forecast_horizons=self.forecast_horizons,
            sample_start_indices=test_indices,
            split=DatasetSplit.TEST,
            feature_names=feature_names,
            target_names=target_names,
            node_ids=compact_data.node_ids,
            start_time_offset=0.0,
            window_seconds=compact_data.window_seconds,
        )

        adjacency = AdjacencyData(
            node_ids=compact_data.node_ids,
            node_to_idx=compact_data.node_to_idx,
            num_nodes=compact_data.num_nodes,
            edge_index=compact_data.edge_index,
            edge_weights=compact_data.edge_weights,
            adjacency_matrix=[list(row) for row in compact_data.adjacency_matrix],
            has_unobserved_gaps={},
        )

        return CompactSpatioTemporalDataset(
            dataset_id=f"ST_DS_{compact_data.dataset_id}",
            input_sequence_length=self.input_sequence_length,
            forecast_horizons=[h[0] for h in self.forecast_horizons],
            feature_names=feature_names,
            target_names=target_names,
            adjacency=adjacency,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            train_dataset=train_split,
            val_dataset=val_split,
            test_dataset=test_split,
            metadata={
                "total_timesteps": T_total,
                "num_nodes": N_nodes,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "normalization_method": self.normalization_method,
                "compact_mode": True,
            },
        )

