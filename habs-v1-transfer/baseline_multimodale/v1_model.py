"""Modality encoders, masked two-second sequences and fold-local preprocessing."""

import numpy as np
import torch
from torch import nn

from .signals import FEATURE_COLUMNS
from .v1_data import EEG, RATES

MODELS = {**{f"v1_{s}": (s,) for s in RATES}, "v1_peripheral": ("ecg", "gsr", "resp"),
          "v1_four_modalities": tuple(RATES), "descriptor_mlp": ("features",)}


class Banks:
    """Keep input arrays on one device; only index selected branches for each batch."""
    def __init__(self, windows, lengths, rows, device):
        self.device = device
        self.x = {s: torch.from_numpy(x).to(device) for s, x in windows.items()}
        self.features = rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self.x["features"] = torch.from_numpy(self.features).to(device)
        self.lengths = {s: torch.tensor(lengths[:, i], device=device) for i, s in enumerate(RATES)}
        self.moments = {}
        for i, (sensor, x) in enumerate(windows.items()):
            # The cache stores zeros outside the valid interval.
            sums = x.sum(axis=2, dtype=np.float64)
            squares = np.einsum("nct,nct->nc", x, x, dtype=np.float64)
            self.moments[sensor] = sums, squares, lengths[:, i]

    def configure(self, model, train):
        if model.sensors == ("features",):
            x = self.features[train].astype(np.float64)
            medians = np.zeros(x.shape[1])
            used = np.isfinite(x).any(axis=0)
            medians[used] = np.nanmedian(x[:, used], axis=0)
            x = np.where(np.isfinite(x), x, medians)
            model.impute.copy_(torch.as_tensor(medians))
            model.feature_used.copy_(torch.as_tensor(used))
            model.mean.copy_(torch.as_tensor(x.mean(axis=0)))
            std = x.std(axis=0)
            model.scale.copy_(torch.as_tensor(np.where(std > 1e-12, std, 1.0)))
            return
        for sensor, branch in model.branches.items():
            sums, squares, counts = self.moments[sensor]
            count = counts[train].sum()
            mean = sums[train].sum(axis=0) / count
            variance = np.maximum(squares[train].sum(axis=0) / count - mean ** 2, 0)
            scale = np.where(variance > 1e-24, np.sqrt(variance), 1)
            branch.mean.copy_(torch.tensor(mean[:, None]))
            branch.scale.copy_(torch.tensor(scale[:, None]))

    def forward(self, model, batch):
        return model({s: self.x[s][batch] for s in model.sensors},
                     {s: self.lengths[s][batch] for s in model.branches})


def valid_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


class Encoder(nn.Module):
    def __init__(self, sensor):
        super().__init__()
        channels = len(EEG) if sensor == "eeg" else 1
        self.register_buffer("mean", torch.zeros(channels, 1))
        self.register_buffer("scale", torch.ones(channels, 1))
        self.sensor = sensor
        if sensor == "eeg":
            self.temporal = nn.Conv2d(1, 4, (1, 25), stride=(1, 5), padding=(0, 12))
            self.spatial = nn.Conv2d(4, 8, (channels, 1), groups=4)
            layers = [(8, 32, 9, 2, 1)]
            initial_stride = 5
        else:
            layers = {
                "ecg": [(1, 8, 25, 5, 1), (8, 16, 9, 2, 1), (16, 32, 7, 1, 1)],
                "gsr": [(1, 8, 9, 1, 1), (8, 32, 13, 1, 2)],
                "resp": [(1, 8, 15, 1, 1), (8, 16, 15, 1, 4), (16, 32, 15, 1, 8)],
            }[sensor]
            initial_stride = 1
        self.layers = nn.ModuleList([
            nn.Conv1d(inputs, outputs, kernel, stride=stride, dilation=dilation,
                      padding=dilation * (kernel // 2))
            for inputs, outputs, kernel, stride, dilation in layers
        ])
        self.bin_size = 2 * RATES[sensor] // (initial_stride * int(np.prod([p[3] for p in layers])))

    def forward(self, x, length):
        x = ((x - self.mean) / self.scale).masked_fill(~valid_mask(length, x.shape[-1])[:, None], 0)
        if self.sensor == "eeg":
            x = torch.nn.functional.elu(self.spatial(self.temporal(x.unsqueeze(1))).squeeze(2))
            length = (length + 4) // 5
            x = x.masked_fill(~valid_mask(length, x.shape[-1])[:, None], 0)
        for layer in self.layers:
            x = torch.nn.functional.elu(layer(x))
            stride = layer.stride[0]
            length = (length + stride - 1) // stride
            x = x.masked_fill(~valid_mask(length, x.shape[-1])[:, None], 0)
        # Bins are anchored at the BDF window start, irrespective of batch padding.
        total = 5 * self.bin_size
        x = torch.nn.functional.pad(x[..., :total], (0, max(0, total - x.shape[-1])))
        counts = (length[:, None] - torch.arange(5, device=x.device) * self.bin_size).clamp(0, self.bin_size)
        tokens = x.reshape(len(x), 32, 5, self.bin_size).sum(-1).transpose(1, 2)
        return tokens / counts.clamp_min(1)[:, :, None], counts > 0


class Regressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.sensors = MODELS[name]
        self.branches = nn.ModuleDict()
        if self.sensors == ("features",):
            width = len(FEATURE_COLUMNS)
            self.register_buffer("mean", torch.zeros(width))
            self.register_buffer("scale", torch.ones(width))
            self.register_buffer("impute", torch.zeros(width))
            self.register_buffer("feature_used", torch.ones(width, dtype=torch.bool))
        else:
            self.branches.update({s: Encoder(s) for s in self.sensors})
            width = len(self.sensors) * 32
        self.head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, inputs, lengths):
        if self.sensors == ("features",):
            x = inputs["features"]
            x = torch.where(torch.isfinite(x), x, self.impute)
            pooled = ((x - self.mean) / self.scale).masked_fill(~self.feature_used, 0)
        else:
            vectors = []
            for sensor, branch in self.branches.items():
                tokens, valid = branch(inputs[sensor], lengths[sensor])
                vectors.append((tokens * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1))
            pooled = torch.cat(vectors, dim=1)
        return self.head(pooled).squeeze(-1)
