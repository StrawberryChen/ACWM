"""Learning-free measurements. This is an AVS-style detector, not optical flow."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def shifted(x, dx, dy):
    """Translate content by (dx,dy), zero boundaries; never wrap around the image."""
    height, width = x.shape[-2:]
    padded = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))
    return padded[..., max(-dy, 0):max(-dy, 0)+height,
                  max(-dx, 0):max(-dx, 0)+width]


class FixedMotionTarget(nn.Module):
    def __init__(self, kind="avs", resolution=6, directions=8, smoothing=True,
                 shifts=(1, 2, 4), projection_dim=288, seed=42, pair_chunk=64):
        super().__init__()
        if kind not in {"avs", "pixel_diff", "random_projection"}:
            raise ValueError(kind)
        if resolution < 1 or directions not in {4, 8} or pair_chunk < 1 or not shifts or min(shifts) < 1:
            raise ValueError("invalid fixed target geometry")
        self.kind, self.resolution, self.directions = kind, resolution, directions
        self.smoothing, self.shifts, self.pair_chunk = smoothing, tuple(shifts), pair_chunk
        self.channels = directions if kind == "avs" else (3 if kind == "pixel_diff" else projection_dim)
        self.map_size = resolution if kind != "random_projection" else 1
        self.output_dim = self.channels*self.map_size**2
        self.spec = dict(kind=kind, resolution=resolution, directions=directions, smoothing=smoothing,
                         shifts=list(shifts), projection_dim=projection_dim, seed=seed)
        t = torch.arange(-2, 3).float()
        kernel = torch.exp(-t.square()/(2*0.8**2))
        kernel /= kernel.sum()
        self.register_buffer("gaussian", (kernel[:, None]*kernel[None]).reshape(1, 1, 5, 5))
        self.register_buffer("rgb_weights", torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1))
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("projection", torch.randn(3*resolution**2, projection_dim, generator=generator)
                             /math.sqrt(3*resolution**2))
        self.register_buffer("mean", torch.zeros(self.channels))
        self.register_buffer("std", torch.ones(self.channels))
        assert not list(self.parameters())

    @torch.no_grad()
    def _avs_pairs(self, previous, following):
        # RGB [N,3,H,W] -> grayscale [N,1,H,W]. Image y increases downwards.
        a, b = (previous*self.rgb_weights).sum(1, keepdim=True), (following*self.rgb_weights).sum(1, keepdim=True)
        if self.smoothing:
            a = F.conv2d(F.pad(a, (2, 2, 2, 2), mode="replicate"), self.gaussian)
            b = F.conv2d(F.pad(b, (2, 2, 2, 2), mode="replicate"), self.gaussian)
        difference = b-a
        temporal_on, temporal_off = difference.relu(), (-difference).relu()
        gate = temporal_on+temporal_off
        # Separate local ON/OFF spatial contrast supports both bright and dark
        # objects. A fixed delayed correlator compares forward vs reverse transport.
        ca = a-F.avg_pool2d(F.pad(a, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        cb = b-F.avg_pool2d(F.pad(b, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
        pa, pb = torch.cat((ca.relu(), (-ca).relu()), 1), torch.cat((cb.relu(), (-cb).relu()), 1)
        offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        if self.directions == 4:
            offsets = offsets[::2]
        outputs = []
        for dx, dy in offsets:
            response = torch.zeros_like(a)
            for distance in self.shifts:
                x, y = dx*distance, dy*distance
                forward = shifted(pa, x, y)*pb
                reverse = pa*shifted(pb, x, y)
                directional = (forward-reverse).sum(1, keepdim=True).relu()
                response += directional*(gate+shifted(gate, x, y))/math.hypot(dx, dy)
            outputs.append(response/len(self.shifts))
        return F.adaptive_avg_pool2d(torch.cat(outputs, 1), self.resolution)

    @torch.no_grad()
    def forward(self, frames, normalize=True):
        # [B,H+1,3,96,96] -> [B,H,D_motion]; transition zero includes X_t -> X_t+1.
        assert frames.ndim == 5 and frames.shape[1] >= 2 and frames.shape[2] == 3
        assert torch.isfinite(frames).all() and frames.min() >= 0 and frames.max() <= 1
        batch, time = frames.shape[:2]
        previous, following = frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1)
        chunks = []
        for start in range(0, len(previous), self.pair_chunk):
            a, b = previous[start:start+self.pair_chunk], following[start:start+self.pair_chunk]
            if self.kind == "avs":
                value = self._avs_pairs(a, b)
            else:
                value = F.adaptive_avg_pool2d(b-a, self.resolution)
                if self.kind == "random_projection":
                    value = (value.flatten(1)@self.projection).reshape(-1, self.channels, 1, 1)
            if normalize:
                value = (value-self.mean[None, :, None, None])/self.std[None, :, None, None]
            chunks.append(value.flatten(1))
        result = torch.cat(chunks).reshape(batch, time-1, self.output_dim)
        assert not result.requires_grad
        return result

    def maps(self, value):
        return value.reshape(*value.shape[:-1], self.channels, self.map_size, self.map_size)

    def denormalize(self, value):
        maps = self.maps(value)
        return (maps*self.std[:, None, None]+self.mean[:, None, None]).flatten(-3)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    def load_statistics(self, path):
        saved = json.loads(Path(path).read_text())
        if saved["fingerprint"] != self.fingerprint():
            raise ValueError("target statistics belong to a different fixed operator")
        self.mean.copy_(torch.tensor(saved["mean"], device=self.mean.device))
        self.std.copy_(torch.tensor(saved["std"], device=self.std.device))
        return saved


@torch.no_grad()
def fit_statistics(operator, loader, device, path, provenance, epsilon=1e-6):
    # Fit UNIQUE adjacent train pairs (horizon=1), not overlapping future windows.
    n = 0
    total = torch.zeros(operator.channels, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    for batch in loader:
        value = operator.maps(operator(batch["frames"].to(device), normalize=False)).double()
        total += value.sum((0, 1, 3, 4))
        square += value.square().sum((0, 1, 3, 4))
        n += value.shape[0]*value.shape[1]*value.shape[3]*value.shape[4]
    if not n:
        raise ValueError("empty train statistics")
    mean = total/n
    std = (square/n-mean.square()).clamp_min(0).sqrt().clamp_min(epsilon)
    operator.mean.copy_(mean.float())
    operator.std.copy_(std.float())
    saved = dict(fingerprint=operator.fingerprint(), operator=operator.spec, count=n,
                 mean=mean.cpu().tolist(), std=std.cpu().tolist(), epsilon=epsilon,
                 train_sources=provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2))
    return saved
