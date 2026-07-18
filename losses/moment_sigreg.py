import torch
from torch import nn


class MomentSIGRegLoss(nn.Module):
    """Isotropic Gaussian moment regularizer for ACWM v3-N1.

    Input shape: means [N, D], where N=2B and D=192.
    """

    def forward(self, means: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert means.ndim == 2, f"MomentSIGReg expects [N,D], got {tuple(means.shape)}"
        count, dim = means.shape
        if count < 2:
            raise ValueError("MomentSIGReg requires at least two samples")
        # batch_mean: [D]
        batch_mean = means.mean(dim=0)
        mean_loss = batch_mean.square().mean()
        # centered: [N,D]
        centered = means - batch_mean
        # covariance: [D,D]
        covariance = centered.T @ centered / (count - 1)
        variance = covariance.diag()
        var_loss = (variance - 1).square().mean()
        off_diag = covariance - torch.diag(variance)
        cov_loss = off_diag.square().sum() / (dim * (dim - 1))
        total = mean_loss + var_loss + cov_loss
        metrics = {
            "loss_sig_mean": mean_loss,
            "loss_sig_var": var_loss,
            "loss_sig_cov": cov_loss,
            "mu_batch_mean_abs": batch_mean.abs().mean(),
            "mu_batch_std_mean": means.std(dim=0).mean(),
            "mu_batch_std_min": means.std(dim=0).min(),
            "mu_batch_std_max": means.std(dim=0).max(),
            "mu_cov_offdiag_abs_mean": off_diag.abs().sum() / (dim * (dim - 1)),
            "mu_cov_diagonal_mean": variance.mean(),
        }
        return total, metrics
