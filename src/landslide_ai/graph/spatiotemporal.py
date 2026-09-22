from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from landslide_ai.graph.gnn import GRAPH_FEATURE_COLUMNS, build_graph_feature_frame, configure_torch_runtime
from landslide_ai.graph.spatial import build_spatial_graph
from landslide_ai.research.temporal_models import EPSILON
from landslide_ai.utils.optional_dependencies import torch_available


SPATIOTEMPORAL_FEATURE_COLUMNS = GRAPH_FEATURE_COLUMNS


@dataclass(slots=True)
class SpatioTemporalDataset:
    sequences: np.ndarray
    labels: np.ndarray
    sample_dates: list[str]
    region_ids: list[str]
    region_metadata: pd.DataFrame
    feature_columns: list[str]
    adjacency: np.ndarray


def build_spatiotemporal_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    sequence_length: int = 7,
    min_regions_per_date: int = 4,
) -> SpatioTemporalDataset:
    if feature_columns is None:
        feature_columns = SPATIOTEMPORAL_FEATURE_COLUMNS

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", "region_id"]).sort_values(["region_id", "date"]).reset_index(drop=True)
    working = build_graph_feature_frame(working)

    region_sequences: dict[str, dict[pd.Timestamp, tuple[np.ndarray, int, pd.Series]]] = {}
    region_metadata_rows: list[pd.Series] = []
    for region_id, group in working.groupby("region_id", dropna=False):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < sequence_length:
            continue
        feature_matrix = group[feature_columns].astype("float32").to_numpy()
        labels = group["label"].astype(int).to_numpy()
        sample_map: dict[pd.Timestamp, tuple[np.ndarray, int, pd.Series]] = {}
        for index in range(sequence_length - 1, len(group)):
            start = index - sequence_length + 1
            sample_map[group.iloc[index]["date"]] = (
                feature_matrix[start : index + 1],
                int(labels[index]),
                group.iloc[index],
            )
        if sample_map:
            region_sequences[str(region_id)] = sample_map
            region_metadata_rows.append(group.iloc[-1])

    if not region_sequences:
        return SpatioTemporalDataset(
            sequences=np.empty((0, 0, sequence_length, len(feature_columns)), dtype="float32"),
            labels=np.empty((0, 0), dtype="float32"),
            sample_dates=[],
            region_ids=[],
            region_metadata=working.iloc[0:0].copy(),
            feature_columns=list(feature_columns),
            adjacency=np.empty((0, 0), dtype="float32"),
        )

    all_dates: dict[pd.Timestamp, list[str]] = {}
    for region_id, sample_map in region_sequences.items():
        for sample_date in sample_map:
            all_dates.setdefault(sample_date, []).append(region_id)

    usable_dates = sorted(date for date, region_ids in all_dates.items() if len(region_ids) >= min_regions_per_date)
    if not usable_dates:
        return SpatioTemporalDataset(
            sequences=np.empty((0, 0, sequence_length, len(feature_columns)), dtype="float32"),
            labels=np.empty((0, 0), dtype="float32"),
            sample_dates=[],
            region_ids=[],
            region_metadata=working.iloc[0:0].copy(),
            feature_columns=list(feature_columns),
            adjacency=np.empty((0, 0), dtype="float32"),
        )

    common_regions = sorted(
        set.intersection(*(set(region_id for region_id in all_dates[date]) for date in usable_dates))
    )
    if len(common_regions) < min_regions_per_date:
        return SpatioTemporalDataset(
            sequences=np.empty((0, 0, sequence_length, len(feature_columns)), dtype="float32"),
            labels=np.empty((0, 0), dtype="float32"),
            sample_dates=[],
            region_ids=[],
            region_metadata=working.iloc[0:0].copy(),
            feature_columns=list(feature_columns),
            adjacency=np.empty((0, 0), dtype="float32"),
        )

    sequence_samples: list[np.ndarray] = []
    label_samples: list[np.ndarray] = []
    sample_dates: list[str] = []
    metadata_lookup = (
        pd.DataFrame(region_metadata_rows)
        .drop_duplicates(subset=["region_id"])
        .set_index("region_id", drop=False)
        .loc[common_regions]
        .reset_index(drop=True)
    )

    for sample_date in usable_dates:
        sequence_samples.append(
            np.asarray([region_sequences[region_id][sample_date][0] for region_id in common_regions], dtype="float32")
        )
        label_samples.append(
            np.asarray([region_sequences[region_id][sample_date][1] for region_id in common_regions], dtype="float32")
        )
        sample_dates.append(pd.Timestamp(sample_date).date().isoformat())

    graph_records = []
    for row in metadata_lookup.to_dict(orient="records"):
        graph_records.append(
            type("GraphRecord", (), row)()
        )
    adjacency = build_spatial_graph(graph_records).normalized_adjacency.astype("float32")

    return SpatioTemporalDataset(
        sequences=np.asarray(sequence_samples, dtype="float32"),
        labels=np.asarray(label_samples, dtype="float32"),
        sample_dates=sample_dates,
        region_ids=list(common_regions),
        region_metadata=metadata_lookup,
        feature_columns=list(feature_columns),
        adjacency=adjacency,
    )


class SpatioTemporalGraphRiskModel:
    def __init__(
        self,
        hidden_size: int = 32,
        dropout: float = 0.15,
        sequence_length: int = 7,
        architecture: str = "gru",
        focal_gamma: float = 1.5,
        use_graph: bool = True,
        use_temporal: bool = True,
        include_vegetation: bool = True,
    ) -> None:
        if not torch_available():
            raise ImportError("PyTorch is not installed. Install `torch` to use the spatio-temporal graph model.")

        configure_torch_runtime()
        import torch
        import torch.nn as nn

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        class _SpatioTemporalNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float) -> None:
                super().__init__()
                self.use_temporal = use_temporal
                self.use_graph = use_graph
                if use_temporal and architecture == "lstm":
                    self.temporal = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
                elif use_temporal:
                    self.temporal = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
                else:
                    self.temporal = nn.Linear(input_dim, hidden_dim)
                self.graph_projection = nn.Linear(hidden_dim, hidden_dim)
                self.local_projection = nn.Linear(hidden_dim, hidden_dim)
                self.dropout = nn.Dropout(dropout_rate)
                self.output = nn.Linear(hidden_dim, 1)

            def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor, keep_dropout: bool = False) -> torch.Tensor:
                batch_size, node_count, sequence_length, feature_count = inputs.shape
                flattened = inputs.reshape(batch_size * node_count, sequence_length, feature_count)
                if self.use_temporal:
                    _, hidden = self.temporal(flattened)
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                    encoded = hidden[-1].reshape(batch_size, node_count, -1)
                else:
                    latest = flattened[:, -1, :]
                    encoded = self.temporal(latest).reshape(batch_size, node_count, -1)
                propagated = torch.einsum("ij,bjh->bih", adjacency, encoded) if self.use_graph else torch.zeros_like(encoded)
                hidden_state = self.local_projection(encoded) + self.graph_projection(propagated)
                hidden_state = torch.relu(hidden_state)
                hidden_state = self.dropout(hidden_state if keep_dropout else hidden_state)
                return self.output(hidden_state).squeeze(-1)

        self.torch = torch
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.architecture = architecture
        self.focal_gamma = focal_gamma
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        self.include_vegetation = include_vegetation
        self._model_class = _SpatioTemporalNet
        self.model = None
        self.feature_columns: list[str] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.region_ids: list[str] = []
        self.region_metadata: pd.DataFrame | None = None

    def _build_model(self, input_dim: int):
        return self._model_class(input_dim=input_dim, hidden_dim=self.hidden_size, dropout_rate=self.dropout)

    def fit(
        self,
        dataset: SpatioTemporalDataset,
        epochs: int = 18,
        learning_rate: float = 0.0015,
        validation_fraction: float = 0.25,
        random_state: int = 42,
        early_stopping_patience: int = 4,
        positive_class_weight_scale: float = 1.0,
        selection_metric: str = "rare_event_score",
    ) -> dict[str, object]:
        torch = self.torch
        self.feature_columns = list(dataset.feature_columns)
        self.region_ids = list(dataset.region_ids)
        self.region_metadata = dataset.region_metadata.copy()
        if len(dataset.sample_dates) == 0:
            raise ValueError("Spatio-temporal dataset is empty. Provide a richer temporal dataset.")

        sequence_tensor = dataset.sequences.astype("float32")
        if not self.include_vegetation:
            vegetation_columns = {"ndvi", "ndvi_loss", "vegetation_vulnerability"}
            vegetation_indices = [
                index for index, column in enumerate(self.feature_columns) if column in vegetation_columns
            ]
            if vegetation_indices:
                sequence_tensor[:, :, :, vegetation_indices] = 0.0
        self.feature_mean = sequence_tensor.mean(axis=(0, 1, 2))
        self.feature_std = np.where(sequence_tensor.std(axis=(0, 1, 2)) == 0.0, 1.0, sequence_tensor.std(axis=(0, 1, 2)))
        normalized_sequences = (sequence_tensor - self.feature_mean.reshape(1, 1, 1, -1)) / self.feature_std.reshape(1, 1, 1, -1)

        sample_count = len(normalized_sequences)
        split_index = max(1, int(round(sample_count * (1.0 - validation_fraction))))
        split_index = min(split_index, sample_count - 1) if sample_count > 1 else sample_count
        train_indices = np.arange(0, split_index)
        validation_indices = np.arange(split_index, sample_count)

        x_train = torch.tensor(normalized_sequences[train_indices], dtype=torch.float32)
        y_train = torch.tensor(dataset.labels[train_indices], dtype=torch.float32)
        adjacency = torch.tensor(dataset.adjacency, dtype=torch.float32)
        self.model = self._build_model(input_dim=len(self.feature_columns))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        positive = float(max(y_train.sum().item(), 1.0))
        negative = float(max(y_train.numel() - y_train.sum().item(), 1.0))
        pos_weight = torch.tensor([max((negative / positive) * positive_class_weight_scale, 1.0)], dtype=torch.float32)

        self.model.train()
        losses: list[float] = []
        best_state = None
        best_val_loss = float("inf")
        best_score = float("-inf")
        best_epoch = 0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_pr_auc = 0.0
        best_roc_auc = 0.0
        patience = 0
        x_val = torch.tensor(normalized_sequences[validation_indices], dtype=torch.float32) if len(validation_indices) > 0 else None
        y_val = torch.tensor(dataset.labels[validation_indices], dtype=torch.float32) if len(validation_indices) > 0 else None
        for epoch_index in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_train, adjacency)
            loss = _focal_bce_loss(logits, y_train, pos_weight=pos_weight, gamma=self.focal_gamma)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if x_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(x_val, adjacency)
                    val_loss = _focal_bce_loss(val_logits, y_val, pos_weight=pos_weight, gamma=self.focal_gamma)
                    val_probabilities = torch.sigmoid(val_logits).cpu().numpy().reshape(-1)
                self.model.train()
                val_loss_value = float(val_loss.item())
                val_labels = y_val.cpu().numpy().reshape(-1).astype(int)
                threshold = _optimal_threshold(val_labels, val_probabilities)
                val_predictions = (val_probabilities >= threshold).astype(int)
                val_pr_auc = (
                    float(average_precision_score(val_labels, val_probabilities))
                    if len(np.unique(val_labels)) > 1
                    else 0.0
                )
                val_roc_auc = (
                    float(roc_auc_score(val_labels, val_probabilities))
                    if len(np.unique(val_labels)) > 1
                    else 0.0
                )
                val_f1 = float(f1_score(val_labels, val_predictions, zero_division=0))
                val_precision = float(precision_score(val_labels, val_predictions, zero_division=0))
                val_recall = float(recall_score(val_labels, val_predictions, zero_division=0))
                selection_score = (
                    _rare_event_score(val_pr_auc, val_f1, val_precision, val_recall, val_roc_auc)
                    if selection_metric == "rare_event_score"
                    else -val_loss_value
                )
                if (
                    selection_score > best_score + 1e-6
                    or (
                        abs(selection_score - best_score) <= 1e-6
                        and val_loss_value + 1e-5 < best_val_loss
                    )
                ):
                    best_score = selection_score
                    best_val_loss = val_loss_value
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    best_epoch = epoch_index + 1
                    best_threshold = threshold
                    best_precision = val_precision
                    best_recall = val_recall
                    best_f1 = val_f1
                    best_pr_auc = val_pr_auc
                    best_roc_auc = val_roc_auc
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        probabilities, uncertainty = self.predict_dataset(dataset, mc_passes=8)
        validation_probs = probabilities[validation_indices].reshape(-1)
        validation_labels = dataset.labels[validation_indices].reshape(-1)
        validation_binary = (validation_probs >= best_threshold).astype(int)
        metrics = {
            "training_loss_final": float(losses[-1]) if losses else 0.0,
            "validation_f1": best_f1 if len(validation_labels) > 0 else 0.0,
            "validation_precision": best_precision if len(validation_labels) > 0 else 0.0,
            "validation_recall": best_recall if len(validation_labels) > 0 else 0.0,
            "validation_threshold": best_threshold,
            "validation_roc_auc": best_roc_auc if len(np.unique(validation_labels.astype(int))) > 1 else 0.0,
            "validation_pr_auc": best_pr_auc if len(np.unique(validation_labels.astype(int))) > 1 else 0.0,
            "validation_uncertainty_mean": float(np.mean(uncertainty[validation_indices])) if len(validation_indices) > 0 else 0.0,
            "validation_indices": validation_indices.tolist(),
            "architecture": self.architecture,
            "focal_gamma": self.focal_gamma,
            "use_graph": self.use_graph,
            "use_temporal": self.use_temporal,
            "include_vegetation": self.include_vegetation,
            "positive_class_weight_scale": positive_class_weight_scale,
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best_selection_score": best_score if best_score != float("-inf") else 0.0,
        }
        return metrics

    def predict_dataset(self, dataset: SpatioTemporalDataset, mc_passes: int = 8) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("SpatioTemporalGraphRiskModel must be fit or loaded before inference.")

        torch = self.torch
        normalized_sequences = (
            dataset.sequences.astype("float32") - self.feature_mean.reshape(1, 1, 1, -1)
        ) / self.feature_std.reshape(1, 1, 1, -1)
        x_tensor = torch.tensor(normalized_sequences, dtype=torch.float32)
        adjacency = torch.tensor(dataset.adjacency, dtype=torch.float32)

        base_probabilities = self._forward_probabilities(x_tensor, adjacency, keep_dropout=False)
        dropout_probabilities = []
        for _ in range(max(mc_passes, 1)):
            dropout_probabilities.append(self._forward_probabilities(x_tensor, adjacency, keep_dropout=True))
        stacked = np.stack(dropout_probabilities, axis=0)
        return base_probabilities, stacked.std(axis=0)

    def _forward_probabilities(self, x_tensor, adjacency_tensor, keep_dropout: bool) -> np.ndarray:
        torch = self.torch
        self.model.train(mode=keep_dropout)
        with torch.no_grad():
            logits = self.model(x_tensor, adjacency_tensor, keep_dropout=keep_dropout)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities.astype("float32")

    def explain_predictions(
        self,
        dataset: SpatioTemporalDataset,
        probabilities: np.ndarray,
        uncertainty: np.ndarray,
        top_k_features: int = 3,
        top_k_neighbors: int = 3,
    ) -> pd.DataFrame:
        if self.region_metadata is None:
            raise RuntimeError("SpatioTemporalGraphRiskModel must be fit or loaded before explanations are available.")

        latest_values = dataset.sequences[:, :, -1, :]
        normalized_latest = (
            latest_values - self.feature_mean.reshape(1, 1, -1)
        ) / self.feature_std.reshape(1, 1, -1)
        mean_abs_feature = np.abs(normalized_latest)
        rows: list[dict[str, object]] = []
        adjacency = dataset.adjacency
        for sample_index, sample_date in enumerate(dataset.sample_dates):
            sample_probabilities = probabilities[sample_index]
            sample_uncertainty = uncertainty[sample_index]
            for node_index, region_id in enumerate(dataset.region_ids):
                feature_scores = mean_abs_feature[sample_index, node_index]
                feature_order = np.argsort(feature_scores)[::-1][:top_k_features]
                top_feature_names = [self.feature_columns[index] for index in feature_order]
                top_feature_weights = [round(float(feature_scores[index]), 4) for index in feature_order]
                neighbor_scores = adjacency[node_index] * sample_probabilities
                neighbor_order = np.argsort(neighbor_scores)[::-1]
                top_neighbors = []
                for neighbor_index in neighbor_order:
                    if neighbor_index == node_index:
                        continue
                    top_neighbors.append(
                        {
                            "region_id": dataset.region_ids[neighbor_index],
                            "weight": round(float(neighbor_scores[neighbor_index]), 4),
                        }
                    )
                    if len(top_neighbors) >= top_k_neighbors:
                        break
                metadata_row = self.region_metadata.iloc[node_index]
                confidence_level = _confidence_level(float(sample_uncertainty[node_index]))
                rows.append(
                    {
                        "region_id": region_id,
                        "state": str(metadata_row["state"]),
                        "district": str(metadata_row["district"]),
                        "date": sample_date,
                        "spatiotemporal_gnn_probability": round(float(sample_probabilities[node_index]), 6),
                        "uncertainty_score": round(float(sample_uncertainty[node_index]), 6),
                        "confidence_level": confidence_level,
                        "recommended_interpretation": _recommended_interpretation(
                            float(sample_probabilities[node_index]),
                            float(sample_uncertainty[node_index]),
                        ),
                        "top_driver": top_feature_names[0] if top_feature_names else "",
                        "top_driver_score": top_feature_weights[0] if top_feature_weights else 0.0,
                        "top_feature_drivers": top_feature_names,
                        "top_feature_driver_scores": top_feature_weights,
                        "top_neighbor_influences": top_neighbors,
                        "sequence_length": self.sequence_length,
                    }
                )
        return pd.DataFrame(rows)

    def save(self, artifact_path: str | Path, metadata: dict[str, object] | None = None) -> None:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Cannot save an untrained spatio-temporal graph model.")

        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout,
                "sequence_length": self.sequence_length,
                "architecture": self.architecture,
                "focal_gamma": self.focal_gamma,
                "use_graph": self.use_graph,
                "use_temporal": self.use_temporal,
                "include_vegetation": self.include_vegetation,
                "region_ids": self.region_ids,
                "region_metadata": None if self.region_metadata is None else self.region_metadata.to_dict(orient="records"),
                "metadata": metadata or {},
            },
            artifact,
        )

    def load(self, artifact_path: str | Path) -> dict[str, object]:
        payload = self.torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
        self.hidden_size = int(payload["hidden_size"])
        self.dropout = float(payload["dropout"])
        self.sequence_length = int(payload["sequence_length"])
        self.architecture = str(payload.get("architecture", self.architecture))
        self.focal_gamma = float(payload.get("focal_gamma", self.focal_gamma))
        self.use_graph = bool(payload.get("use_graph", self.use_graph))
        self.use_temporal = bool(payload.get("use_temporal", self.use_temporal))
        self.include_vegetation = bool(payload.get("include_vegetation", self.include_vegetation))
        self.feature_columns = list(payload["feature_columns"])
        self.feature_mean = np.asarray(payload["feature_mean"], dtype="float32")
        self.feature_std = np.asarray(payload["feature_std"], dtype="float32")
        self.region_ids = list(payload.get("region_ids", []))
        region_metadata = payload.get("region_metadata", [])
        self.region_metadata = pd.DataFrame(region_metadata)
        self.model = self._build_model(input_dim=len(self.feature_columns))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return dict(payload.get("metadata", {}))


def run_spatiotemporal_inference_from_frame(
    frame: pd.DataFrame,
    artifact_path: str | Path,
    sequence_length: int = 7,
) -> pd.DataFrame:
    dataset = build_spatiotemporal_dataset(frame, sequence_length=sequence_length)
    if len(dataset.sample_dates) == 0:
        return pd.DataFrame()
    model = SpatioTemporalGraphRiskModel(sequence_length=sequence_length)
    metadata = model.load(artifact_path)
    probabilities, uncertainty = model.predict_dataset(dataset)
    explanation_frame = model.explain_predictions(dataset, probabilities, uncertainty)
    if metadata:
        explanation_frame["artifact_metadata_model_type"] = metadata.get("model_type", "")
    return explanation_frame


def _confidence_level(uncertainty_score: float) -> str:
    if uncertainty_score <= 0.04:
        return "High"
    if uncertainty_score <= 0.09:
        return "Medium"
    return "Low"


def _recommended_interpretation(probability: float, uncertainty_score: float) -> str:
    confidence_level = _confidence_level(uncertainty_score)
    if probability >= 0.70 and confidence_level == "High":
        return "Escalate monitoring and field verification."
    if probability >= 0.70:
        return "High predicted risk, but confirm with supporting evidence."
    if probability >= 0.45:
        return "Moderate signal. Watch rainfall persistence and neighboring districts."
    if confidence_level == "Low":
        return "Low current score with weak confidence. Recheck after the next data refresh."
    return "Low current score. Continue routine monitoring."


def _focal_bce_loss(logits, targets, pos_weight, gamma: float):
    import torch

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    probabilities = torch.sigmoid(logits)
    pt = torch.where(targets >= 0.5, probabilities, 1.0 - probabilities).clamp(EPSILON, 1.0 - EPSILON)
    focal_term = (1.0 - pt) ** gamma
    return (focal_term * bce).mean()


def _optimal_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    scores = []
    for index, threshold in enumerate(thresholds):
        p = float(precision[index])
        r = float(recall[index])
        f1 = (2.0 * p * r) / max(p + r, EPSILON)
        scores.append(f1 + (0.4 * p) + (0.25 * r))
    best_index = int(np.nanargmax(np.asarray(scores)))
    return float(np.clip(thresholds[best_index], 0.10, 0.90))


def _rare_event_score(pr_auc: float, f1: float, precision: float, recall: float, roc_auc: float) -> float:
    return (1.6 * pr_auc) + (1.2 * f1) + (0.7 * precision) + (0.35 * recall) + (0.45 * roc_auc)
