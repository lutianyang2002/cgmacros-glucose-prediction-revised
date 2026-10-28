"""Shared compact GRU/TCN models and fixed-budget training.

The module does not execute notebooks or allow TEST data. Training supports validation early stopping or an explicit fixed-epoch budget."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch


PROTOCOLS = {"A": "Within-subject Future Prediction", "B": "Unseen-subject Generalization"}
KEYS = ["evaluation_protocol", "subject", "origin_time", "task_id"]


class CompactGRU(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = torch.nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class CausalBlock(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = torch.nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.skip = torch.nn.Conv1d(c_in, c_out, 1) if c_in != c_out else torch.nn.Identity()

    def forward(self, x):
        # Left padding only: the output at step s depends on inputs at steps <= s.
        return torch.relu(self.conv(torch.nn.functional.pad(x, (self.pad, 0)))) + self.skip(x)


class CompactTCN(torch.nn.Module):
    def __init__(self, channels: int, kernel: int, dilations):
        super().__init__()
        self.blocks = torch.nn.Sequential(*[CausalBlock(1 if i == 0 else channels, channels, kernel, d) for i, d in enumerate(dilations)])
        self.head = torch.nn.Linear(channels, 1)

    def forward(self, x):  # x: (batch, steps, 1)
        return self.head(self.blocks(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


def make_model(family: str):
    if family == "GRU":
        return CompactGRU(32, 1)
    if family == "TCN":
        return CompactTCN(16, 3, [1, 2, 4, 8])
    raise ValueError(f"unknown network {family}")


def require_development(frame):
    if len(frame) == 0 or not set(frame["split"]) <= {"train", "validation"}:
        raise ValueError("only nonempty TRAIN/VALIDATION objects are accepted")
    if frame.duplicated(KEYS).any():
        raise ValueError("duplicate development window keys")


def training_statistics(frame, history):
    require_development(frame)
    if not frame.split.eq("train").all():
        raise ValueError("statistics must be estimated on TRAIN only")
    return {"history_mean": history.mean(), "history_sd": history.std(),
            "residual_sd": (frame.target_value.to_numpy() - history[:, -1]).std()}


def run_model(model, x_std, device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.as_tensor(x_std[i:i + 8192, :, None], dtype=torch.float32, device=device))
                          for i in range(0, len(x_std), 8192)]).cpu().numpy()


def metrics(frame, prediction):
    require_development(frame)
    error = pd.Series(np.asarray(prediction) - frame.target_value.to_numpy())
    subject = frame.subject.reset_index(drop=True)
    per_subject = error.groupby(subject).agg(mae=lambda e: e.abs().mean(), rmse=lambda e: np.sqrt((e ** 2).mean()))
    return {"subject_equal_rmse": float(per_subject.rmse.mean()), "pooled_window_rmse": float(np.sqrt((error ** 2).mean())),
            "subject_equal_mae": float(per_subject.mae.mean()), "n_windows": len(frame), "n_subjects": frame.subject.nunique()}


def train_one(family, train, validation, h_train, h_val, *, seed=0, device="cpu", epochs=60,
              early_stopping=True, deadline=None):
    """Sequence-model training with early stopping or a fixed epoch budget.

    Validation is not evaluated during fixed-epoch training. No mixed precision,
    optimizer changes, batch-size changes or new randomness is introduced.
    """
    require_development(validation)
    stats = training_statistics(train, h_train)
    x_train = torch.as_tensor(((h_train - stats["history_mean"]) / stats["history_sd"])[:, :, None], dtype=torch.float32, device=device)
    y_train = torch.as_tensor((train.target_value.to_numpy() - h_train[:, -1]) / stats["residual_sd"], dtype=torch.float32, device=device)
    x_val_std = (h_val - stats["history_mean"]) / stats["history_sd"]
    y_val = (validation.target_value.to_numpy() - h_val[:, -1]) / stats["residual_sd"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(family).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, best_state, wait = np.inf, 0, None, 0
    start = time.perf_counter()
    epoch_records = []
    for epoch in range(1, epochs + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("sequence-model benchmark 900-second compute budget exhausted")
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(y_train), generator=generator).to(device)
        for i in range(0, len(order), 512):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("sequence-model benchmark 900-second compute budget exhausted")
            idx = order[i:i + 512]
            optimizer.zero_grad()
            loss = torch.mean((model(x_train[idx]) - y_train[idx]) ** 2)
            loss.backward()
            optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start
        validation_seconds = 0.0
        if early_stopping:
            validation_start = time.perf_counter()
            val_mse = float(np.mean((run_model(model, x_val_std, device) - y_val) ** 2))
            validation_seconds = time.perf_counter() - validation_start
            if val_mse < best:
                best, best_epoch, wait = val_mse, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
        epoch_records.append({"epoch": epoch, "train_seconds": train_seconds, "validation_seconds": validation_seconds})
        if early_stopping and wait >= 6:
            break
    if early_stopping:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    prediction = h_val[:, -1] + run_model(model, x_val_std, device) * stats["residual_sd"]
    record = {"family": family, "seed": seed, "device": device, "epochs_run": epoch, "best_epoch": best_epoch,
              "best_validation_mse_standardized": best if early_stopping else None,
              "total_seconds": time.perf_counter() - start, "train_windows": len(train), "validation_windows": len(validation),
              "early_stopping": early_stopping, "statistics": {k: float(v) for k, v in stats.items()},
              "metrics": metrics(validation, prediction), "epoch_records": epoch_records,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return record, prediction, state
