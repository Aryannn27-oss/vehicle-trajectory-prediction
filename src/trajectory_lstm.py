import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):

    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


class TrajectoryPredictor:

    def __init__(self, device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TrajectoryLSTM().to(self.device)

        self.model.eval()

      
        self.min_points = 8

    def prepare_sequence(self, trajectory):

        seq = torch.tensor(trajectory, dtype=torch.float32)

        seq = seq.unsqueeze(0)

        return seq.to(self.device)

    def fallback_prediction(self, trajectory):

        if len(trajectory) < 2:
            return None

        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]

        px = x2 + (x2 - x1)
        py = y2 + (y2 - y1)

        return int(px), int(py)

    def predict(self, trajectory):

     if len(trajectory) < self.min_points:
        return None

     try:

        seq = self.prepare_sequence(trajectory[-self.min_points:])

        with torch.no_grad():
            pred = self.model(seq)

        px = int(pred[0, 0].item())
        py = int(pred[0, 1].item())

        return px, py

     except Exception:

        if len(trajectory) < 2:
            return None

        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]

        vx = x2 - x1
        vy = y2 - y1

        pred_x = x2 + vx
        pred_y = y2 + vy

        return int(pred_x), int(pred_y)
