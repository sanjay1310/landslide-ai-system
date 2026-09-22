from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from landslide_ai.data.loader import load_region_records, load_training_frame
from landslide_ai.graph.spatial import build_spatial_graph
from landslide_ai.utils.optional_dependencies import torch_available


GRAPH_FEATURE_COLUMNS = [
    "rainfall_24h_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_14d_mm",
    "rainfall_ratio_24h_to_7d",
    "rainfall_acceleration",
    "rainfall_3d_to_14d_ratio",
    "rainfall_7d_to_14d_ratio",
    "rolling_intensity_gap",
    "rainfall_intensity_x_acceleration",
    "slope_deg",
    "elevation_m",
    "elevation_slope_interaction",
    "soil_wetness_index",
    "slope_soil_interaction",
    "ndvi",
    "ndvi_loss",
    "vegetation_vulnerability",
    "temperature_c",
    "temperature_rainfall_interaction",
    "neighbor_risk_mean",
    "neighbor_count",
    "spatial_neighbor_load",
    "neighbor_pressure",
    "terrain_rainfall_interaction",
    "soil_rainfall_interaction",
    "terrain_ruggedness",
    "drainage_density",
    "road_density_km_km2",
    "river_distance_km",
    "geology_risk_index",
    "land_cover_risk_index",
    "historical_landslide_frequency",
    "day_of_year_sin",
    "day_of_year_cos",
]
@dataclass(slots=True)
class GraphModelArtifacts:
    artifact_path: Path


def configure_torch_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


class GraphNeuralRiskModel:
    def __init__(self, hidden_dim: int = 24) -> None:
        if not torch_available():
            raise ImportError("PyTorch is not installed. Install `torch` to use the graph model.")

        configure_torch_runtime()
        import torch
        import torch.nn as nn

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        if hasattr(torch, "multiprocessing"):
            try:
                torch.multiprocessing.set_sharing_strategy("file_system")
            except (AttributeError, RuntimeError):
                pass

        self.torch = torch
        self.nn = nn
        self.hidden_dim = hidden_dim
        self.model = None
        self.feature_columns: list[str] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def _build_model(self, input_dim: int):
        nn = self.nn
        torch = self.torch
        hidden_dim = self.hidden_dim

        class _GCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Linear(input_dim, hidden_dim, bias=False)
                self.conv2 = nn.Linear(hidden_dim, 1, bias=False)

            def forward(self, x_tensor, adjacency_tensor):
                hidden = adjacency_tensor @ x_tensor
                hidden = self.conv1(hidden)
                hidden = torch.relu(hidden)
                hidden = adjacency_tensor @ hidden
                logits = self.conv2(hidden).squeeze(-1)
                return logits

        return _GCN()

    def fit(
        self,
        frame: pd.DataFrame,
        adjacency: np.ndarray,
        label_column: str = "label",
        feature_columns: list[str] | None = None,
        epochs: int = 300,
        learning_rate: float = 0.02,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.25,
        random_state: int = 42,
        train_indices: list[int] | np.ndarray | None = None,
        validation_indices: list[int] | np.ndarray | None = None,
    ) -> dict[str, object]:
        torch = self.torch
        if feature_columns is None:
            feature_columns = GRAPH_FEATURE_COLUMNS
        self.feature_columns = feature_columns

        feature_frame = build_graph_feature_frame(frame)
        features = feature_frame[feature_columns].astype("float32").to_numpy()
        labels = frame[label_column].astype("float32").to_numpy()
        self.feature_mean = features.mean(axis=0)
        self.feature_std = np.where(features.std(axis=0) == 0.0, 1.0, features.std(axis=0))
        scaled = (features - self.feature_mean) / self.feature_std

        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32)
        adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)

        self.model = self._build_model(input_dim=len(feature_columns))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if train_indices is None or validation_indices is None:
            indices = np.arange(len(frame))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=validation_fraction,
                random_state=random_state,
                stratify=labels.astype(int),
            )
        else:
            train_idx = np.asarray(train_indices, dtype=int)
            val_idx = np.asarray(validation_indices, dtype=int)
        positive_count = float(max(labels[train_idx].sum(), 1.0))
        negative_count = float(max(len(train_idx) - labels[train_idx].sum(), 1.0))
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
        criterion = self.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        train_mask = torch.zeros(len(frame), dtype=torch.bool)
        val_mask = torch.zeros(len(frame), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        losses: list[float] = []
        val_losses: list[float] = []
        best_state = None
        best_val_loss = float("inf")
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(x_tensor, adjacency_tensor)
            loss = criterion(logits[train_mask], y_tensor[train_mask])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x_tensor, adjacency_tensor)
                if val_mask.any():
                    val_loss = criterion(val_logits[val_mask], y_tensor[val_mask])
                else:
                    val_loss = criterion(val_logits[train_mask], y_tensor[train_mask])
                val_losses.append(float(val_loss.item()))
                if float(val_loss.item()) < best_val_loss:
                    best_val_loss = float(val_loss.item())
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            self.model.train()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        probabilities = self.predict_proba_from_frame(frame, adjacency)
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = float(accuracy_score(labels.astype(int), predictions))
        if len(val_idx) > 0:
            val_accuracy = float(accuracy_score(labels[val_idx].astype(int), predictions[val_idx]))
            val_f1 = float(f1_score(labels[val_idx].astype(int), predictions[val_idx], zero_division=0))
        else:
            val_accuracy = accuracy
            val_f1 = float(f1_score(labels.astype(int), predictions, zero_division=0))
        return {
            "losses": losses,
            "val_losses": val_losses,
            "probabilities": probabilities,
            "accuracy": accuracy,
            "validation_accuracy": val_accuracy,
            "validation_f1": val_f1,
            "validation_indices": val_idx.tolist(),
        }

    def predict_proba_from_frame(self, frame: pd.DataFrame, adjacency: np.ndarray) -> np.ndarray:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("GraphNeuralRiskModel must be fit or loaded before inference.")

        torch = self.torch
        feature_frame = build_graph_feature_frame(frame)
        features = feature_frame[self.feature_columns].astype("float32").to_numpy()
        scaled = (features - self.feature_mean) / self.feature_std
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor, adjacency_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return probabilities.astype(float)

    def save(self, artifact_path: str | Path) -> None:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Cannot save an untrained graph model.")

        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "hidden_dim": self.hidden_dim,
            },
            artifact,
        )

    def load(self, artifact_path: str | Path) -> None:
        artifact = Path(artifact_path)
        payload = self.torch.load(artifact, map_location="cpu", weights_only=False)
        self.hidden_dim = int(payload["hidden_dim"])
        self.feature_columns = list(payload["feature_columns"])
        self.feature_mean = np.asarray(payload["feature_mean"], dtype="float32")
        self.feature_std = np.asarray(payload["feature_std"], dtype="float32")
        self.model = self._build_model(input_dim=len(self.feature_columns))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()


def train_graph_model_from_csv(
    csv_path: str | Path,
    artifact_path: str | Path,
    epochs: int = 300,
    learning_rate: float = 0.02,
    validation_fraction: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    frame = load_training_frame(csv_path)
    records = load_region_records(csv_path)
    graph = build_spatial_graph(records)
    model = GraphNeuralRiskModel()
    result = model.fit(
        frame=frame,
        adjacency=graph.normalized_adjacency,
        epochs=epochs,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    model.save(artifact_path)
    return {
        "artifact_path": str(artifact_path),
        "feature_columns": model.feature_columns,
        "probabilities": result["probabilities"],
        "accuracy": result["accuracy"],
        "validation_accuracy": result["validation_accuracy"],
        "validation_f1": result["validation_f1"],
        "validation_indices": result["validation_indices"],
        "losses": result["losses"],
        "val_losses": result["val_losses"],
    }


def predict_graph_probabilities_for_csv(
    csv_path: str | Path,
    artifact_path: str | Path,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    records = load_region_records(csv_path)
    graph = build_spatial_graph(records)
    model = GraphNeuralRiskModel()
    model.load(artifact_path)
    probabilities = model.predict_proba_from_frame(frame, graph.normalized_adjacency)
    output = frame.copy()
    output["graph_gnn_probability"] = probabilities
    return output


def build_graph_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = frame.copy()
    if "rainfall_3d_mm" not in feature_frame.columns:
        rainfall_24h_seed = pd.to_numeric(feature_frame.get("rainfall_24h_mm", 0.0), errors="coerce").fillna(0.0)
        feature_frame["rainfall_3d_mm"] = rainfall_24h_seed * 2.0
    if "rainfall_14d_mm" not in feature_frame.columns:
        rainfall_7d_seed = pd.to_numeric(feature_frame.get("rainfall_7d_mm", 0.0), errors="coerce").fillna(0.0)
        feature_frame["rainfall_14d_mm"] = rainfall_7d_seed * 1.8
    if "rainfall_acceleration" not in feature_frame.columns:
        rainfall_24h_seed = pd.to_numeric(feature_frame.get("rainfall_24h_mm", 0.0), errors="coerce").fillna(0.0)
        if "date" in feature_frame.columns and "region_id" in feature_frame.columns:
            working = feature_frame.copy()
            working["date"] = pd.to_datetime(working["date"], errors="coerce")
            working = working.sort_values(["region_id", "date"]).reset_index()
            accel = rainfall_24h_seed.reindex(working["index"]).reset_index(drop=True) - (
                rainfall_24h_seed.reindex(working["index"]).reset_index(drop=True).groupby(working["region_id"]).shift(1).fillna(0.0)
            )
            working["rainfall_acceleration"] = accel
            feature_frame["rainfall_acceleration"] = working.sort_values("index")["rainfall_acceleration"].to_numpy()
        else:
            feature_frame["rainfall_acceleration"] = 0.0
    if "vegetation_vulnerability" not in feature_frame.columns:
        ndvi_seed = pd.to_numeric(feature_frame.get("ndvi", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        soil_seed = pd.to_numeric(feature_frame.get("soil_wetness_index", 0.0), errors="coerce").fillna(0.0)
        feature_frame["vegetation_vulnerability"] = (1.0 - ndvi_seed) * soil_seed
    if "terrain_rainfall_interaction" not in feature_frame.columns:
        feature_frame["terrain_rainfall_interaction"] = (
            pd.to_numeric(feature_frame.get("slope_deg", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(feature_frame.get("rainfall_24h_mm", 0.0), errors="coerce").fillna(0.0)
        )
    if "soil_rainfall_interaction" not in feature_frame.columns:
        feature_frame["soil_rainfall_interaction"] = (
            pd.to_numeric(feature_frame.get("soil_wetness_index", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(feature_frame.get("rainfall_7d_mm", 0.0), errors="coerce").fillna(0.0)
        )

    rainfall_24h = feature_frame["rainfall_24h_mm"].astype(float)
    rainfall_3d = feature_frame["rainfall_3d_mm"].astype(float)
    rainfall_7d = feature_frame["rainfall_7d_mm"].astype(float)
    rainfall_14d = feature_frame["rainfall_14d_mm"].astype(float).clip(lower=1.0)
    slope = feature_frame["slope_deg"].astype(float)
    elevation = feature_frame["elevation_m"].astype(float)
    soil = feature_frame["soil_wetness_index"].astype(float)
    ndvi = feature_frame["ndvi"].astype(float).clip(0.0, 1.0)
    temperature = feature_frame["temperature_c"].astype(float)
    neighbor_risk = feature_frame["neighbor_risk_mean"].astype(float).clip(0.0, 1.0)
    neighbor_count = feature_frame["neighbor_count"].astype(float)

    feature_frame["rainfall_ratio_24h_to_7d"] = rainfall_24h / np.maximum(rainfall_7d, 1.0)
    feature_frame["rainfall_3d_to_14d_ratio"] = rainfall_3d / rainfall_14d
    feature_frame["rainfall_7d_to_14d_ratio"] = rainfall_7d / rainfall_14d
    feature_frame["rolling_intensity_gap"] = rainfall_24h - (rainfall_7d / 7.0)
    feature_frame["rainfall_intensity_x_acceleration"] = rainfall_24h * feature_frame["rainfall_acceleration"].astype(float).clip(lower=0.0)
    feature_frame["elevation_slope_interaction"] = elevation * slope
    feature_frame["slope_soil_interaction"] = slope * soil
    feature_frame["ndvi_loss"] = 1.0 - ndvi
    feature_frame["vegetation_vulnerability"] = (1.0 - ndvi) * soil
    feature_frame["spatial_neighbor_load"] = neighbor_risk * np.log1p(neighbor_count)
    feature_frame["neighbor_pressure"] = neighbor_risk * (0.5 + (neighbor_count.clip(0.0, 8.0) / 8.0) * 0.5)
    feature_frame["temperature_rainfall_interaction"] = temperature * rainfall_3d
    feature_frame["terrain_rainfall_interaction"] = slope * rainfall_24h
    feature_frame["soil_rainfall_interaction"] = soil * rainfall_7d
    feature_frame["terrain_ruggedness"] = np.sqrt(np.maximum(elevation, 0.0) * np.maximum(slope, 0.0))
    missing_series = pd.Series(np.nan, index=feature_frame.index, dtype="float64")
    feature_frame["drainage_density"] = (
        pd.to_numeric(feature_frame.get("drainage_density", missing_series), errors="coerce")
        .fillna((rainfall_7d / np.maximum(slope + 1.0, 1.0)).clip(0.0, 25.0))
    )
    feature_frame["road_density_km_km2"] = (
        pd.to_numeric(feature_frame.get("road_density_km_km2", missing_series), errors="coerce")
        .fillna((neighbor_count / np.maximum(1.0 + slope / 12.0, 1.0)).clip(0.0, 15.0))
    )
    feature_frame["river_distance_km"] = (
        pd.to_numeric(feature_frame.get("river_distance_km", missing_series), errors="coerce")
        .fillna((10.0 - np.minimum(rainfall_24h / 15.0, 9.0)).clip(0.5, 10.0))
    )
    feature_frame["geology_risk_index"] = (
        pd.to_numeric(feature_frame.get("geology_risk_index", missing_series), errors="coerce")
        .fillna(((slope / 50.0) * 0.6 + (soil.clip(0.0, 1.0) * 0.4)).clip(0.0, 1.0))
    )
    feature_frame["land_cover_risk_index"] = (
        pd.to_numeric(feature_frame.get("land_cover_risk_index", missing_series), errors="coerce")
        .fillna(((1.0 - ndvi) * 0.7 + neighbor_risk * 0.3).clip(0.0, 1.0))
    )
    if "historical_landslide_frequency" not in feature_frame.columns:
        if "label" in feature_frame.columns and "date" in feature_frame.columns and "region_id" in feature_frame.columns:
            working = feature_frame.copy()
            working["date"] = pd.to_datetime(working["date"], errors="coerce")
            working = working.sort_values(["region_id", "date"]).reset_index()
            rolling = (
                working.groupby("region_id")["label"]
                .transform(lambda values: values.shift(1).fillna(0).rolling(window=5, min_periods=1).mean())
                .clip(0.0, 1.0)
            )
            feature_frame["historical_landslide_frequency"] = working.assign(historical_landslide_frequency=rolling).sort_values("index")["historical_landslide_frequency"].to_numpy()
        else:
            feature_frame["historical_landslide_frequency"] = 0.0

    if "date" in feature_frame.columns:
        date_series = pd.to_datetime(feature_frame["date"], errors="coerce")
        day_of_year = date_series.dt.dayofyear.fillna(1).astype(float)
    else:
        day_of_year = pd.Series(np.ones(len(feature_frame)), index=feature_frame.index, dtype="float64")
    feature_frame["day_of_year_sin"] = np.sin((2.0 * np.pi * day_of_year) / 365.0)
    feature_frame["day_of_year_cos"] = np.cos((2.0 * np.pi * day_of_year) / 365.0)
    return feature_frame
