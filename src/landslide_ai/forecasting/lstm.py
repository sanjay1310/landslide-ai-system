from __future__ import annotations

from dataclasses import dataclass

from landslide_ai.utils.optional_dependencies import torch_available


@dataclass(slots=True)
class TorchImportErrorInfo:
    message: str


class RainfallLSTMForecaster:
    """Small LSTM regressor with lazy torch imports."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1) -> None:
        if not torch_available():
            raise ImportError(
                "PyTorch is not installed. Install `torch` to use the LSTM forecaster."
            )

        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.output = nn.Linear(hidden_size, 1)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                sequence_output, _ = self.lstm(features)
                last_step = sequence_output[:, -1, :]
                return self.output(last_step)

        self.torch = torch
        self.model = _Model()

    def fit(
        self,
        x_train,
        y_train,
        epochs: int = 250,
        learning_rate: float = 0.01,
    ) -> list[float]:
        torch = self.torch
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        losses: list[float] = []
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(x_train).squeeze(-1)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        return losses

    def predict(self, x_input):
        self.model.eval()
        with self.torch.no_grad():
            return self.model(x_input).squeeze(-1)
