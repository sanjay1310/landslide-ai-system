from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from landslide_ai.utils.optional_dependencies import torch_available


EPSILON = 1e-6


@dataclass(slots=True)
class ProbabilityCalibrator:
    method: str
    model: object

    def transform(self, probabilities: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float).clip(EPSILON, 1.0 - EPSILON)
        if self.method == "sigmoid":
            logits = np.log(values / (1.0 - values)).reshape(-1, 1)
            calibrated = self.model.predict_proba(logits)[:, 1]
            return np.asarray(calibrated, dtype=float).clip(0.0, 1.0)
        calibrated = self.model.predict(values)
        return np.asarray(calibrated, dtype=float).clip(0.0, 1.0)


def fit_probability_calibrator(
    y_true: pd.Series | list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    method: str = "sigmoid",
) -> ProbabilityCalibrator | None:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float).clip(EPSILON, 1.0 - EPSILON)
    if len(np.unique(labels)) < 2:
        return None

    if method == "sigmoid":
        logits = np.log(values / (1.0 - values)).reshape(-1, 1)
        calibrator = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        calibrator.fit(logits, labels)
        return ProbabilityCalibrator(method="sigmoid", model=calibrator)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(values, labels)
    return ProbabilityCalibrator(method="isotonic", model=calibrator)


@dataclass(slots=True)
class SequenceDatasetBundle:
    sequences: np.ndarray
    labels: np.ndarray
    rows: pd.DataFrame


def build_temporal_sequence_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 7,
) -> SequenceDatasetBundle:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.sort_values(["region_id", "date"]).reset_index(drop=True)

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[pd.Series] = []

    for _, group in working.groupby("region_id", dropna=False):
        if len(group) < sequence_length:
            continue
        feature_matrix = group[feature_columns].astype("float32").to_numpy()
        group_labels = group["label"].astype(int).to_numpy()
        for index in range(sequence_length - 1, len(group)):
            start = index - sequence_length + 1
            sequences.append(feature_matrix[start : index + 1])
            labels.append(int(group_labels[index]))
            rows.append(group.iloc[index])

    if not sequences:
        return SequenceDatasetBundle(
            sequences=np.empty((0, sequence_length, len(feature_columns)), dtype="float32"),
            labels=np.empty((0,), dtype="int64"),
            rows=working.iloc[0:0].copy(),
        )

    return SequenceDatasetBundle(
        sequences=np.asarray(sequences, dtype="float32"),
        labels=np.asarray(labels, dtype="int64"),
        rows=pd.DataFrame(rows).reset_index(drop=True),
    )


class TemporalSequenceRiskClassifier:
    """Small recurrent classifier for district-date risk sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.10,
        architecture: str = "gru",
    ) -> None:
        if not torch_available():
            raise ImportError("PyTorch is not installed. Install `torch` to use temporal sequence models.")

        import torch
        import torch.nn as nn

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        class _TemporalConvBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                encoded = self.network(features.transpose(1, 2)).squeeze(-1)
                return encoded

        class _TransformerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_size, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4 if hidden_size % 4 == 0 else 2,
                    dim_feedforward=max(hidden_size * 2, 32),
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(num_layers, 1))

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                encoded = self.input_projection(features)
                encoded = self.encoder(encoded)
                return encoded[:, -1, :]

        class _SequenceModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.architecture = architecture
                if architecture == "lstm":
                    self.recurrent = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=recurrent_dropout,
                        batch_first=True,
                    )
                elif architecture == "tcn":
                    self.recurrent = _TemporalConvBlock()
                elif architecture == "transformer":
                    self.recurrent = _TransformerBlock()
                else:
                    self.recurrent = nn.GRU(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=recurrent_dropout,
                        batch_first=True,
                    )
                self.dropout = nn.Dropout(dropout)
                self.output = nn.Linear(hidden_size, 1)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                if self.architecture in {"tcn", "transformer"}:
                    encoded = self.recurrent(features)
                    last_step = self.dropout(encoded)
                else:
                    sequence_output, _ = self.recurrent(features)
                    last_step = self.dropout(sequence_output[:, -1, :])
                return self.output(last_step).squeeze(-1)

        self.architecture = architecture
        self.torch = torch
        self.model = _SequenceModel()

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 12,
        learning_rate: float = 0.001,
        batch_size: int = 256,
    ) -> list[float]:
        torch = self.torch
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        positive_count = float(max(y_train.sum(), 1))
        negative_count = float(max(len(y_train) - y_train.sum(), 1))
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        losses: list[float] = []
        self.model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
            losses.append(epoch_loss / max(batch_count, 1))
        return losses

    def predict_proba(self, x_input: np.ndarray, batch_size: int = 512) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        x_tensor = torch.tensor(x_input, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(x_tensor, batch_size=batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x in loader:
                logits = self.model(batch_x)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                outputs.append(probabilities)
        if not outputs:
            return np.empty((0,), dtype="float32")
        return np.concatenate(outputs).astype("float32")
