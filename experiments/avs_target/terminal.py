"""Fixed terminal measurements shared verbatim by targets and goal images."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FixedTerminalOperator(nn.Module):
    def __init__(self, backend="downsample", size=24, dct_coefficients=8):
        super().__init__()
        if backend not in {"downsample", "dct"}:
            raise ValueError("terminal backend must be downsample or dct")
        if size < 1 or not 1 <= dct_coefficients <= size:
            raise ValueError("invalid terminal measurement geometry")
        self.backend, self.size, self.dct_coefficients = backend, size, dct_coefficients
        self.output_dim = 3*(size if backend == "downsample" else dct_coefficients)**2
        self.spec = dict(backend=backend, size=size, dct_coefficients=dct_coefficients,
                         interpolation="area", normalization="rgb_0_1_dct_divide_by_size")
        n, k = torch.arange(size).float(), torch.arange(size).float()[:, None]
        basis = math.sqrt(2/size)*torch.cos(math.pi*(n+0.5)*k/size)
        basis[0] = math.sqrt(1/size)
        self.register_buffer("dct_basis", basis)
        assert not list(self.parameters())

    @torch.no_grad()
    def forward(self, image):
        # RGB [B,3,96,96] -> fixed measurement [B,D_terminal]. No learned norm.
        assert image.ndim == 4 and image.shape[1] == 3
        assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        small = F.interpolate(image.float(), (self.size, self.size), mode="area")
        if self.backend == "downsample":
            return small.flatten(1)
        coefficients = self.dct_basis@small@self.dct_basis.T
        c = self.dct_coefficients
        return (coefficients[..., :c, :c]/self.size).flatten(1)

    @torch.no_grad()
    def image(self, measurement):
        # Visualization only; no image reconstruction objective. DCT is truncated.
        assert measurement.ndim == 2 and measurement.shape[-1] == self.output_dim
        if self.backend == "downsample":
            return measurement.reshape(-1, 3, self.size, self.size)
        coefficients = measurement.new_zeros(len(measurement), 3, self.size, self.size)
        c = self.dct_coefficients
        coefficients[..., :c, :c] = measurement.reshape(-1, 3, c, c)*self.size
        return self.dct_basis.T@coefficients@self.dct_basis
