import torch
from torch import nn
from torch.nn import functional as F


def reference_progress(velocities, eps=1e-8):
    assert velocities.ndim == 3  # [B,K,D]
    reference = velocities[:, 0]  # [B,D], norm PER SAMPLE, not batch-wide
    direction = (reference/(reference.norm(dim=-1, keepdim=True)+eps)).detach()
    return (velocities*direction[:, None]).sum(-1)  # [B,K]; ONLY d detached


def output_progress(output, eps=1e-8):
    if "direction" not in output:
        return reference_progress(output["velocities"], eps)
    velocity, direction = output["velocities"], output["direction"]
    assert velocity.ndim == 4
    if direction.ndim == 3:
        assert direction.shape == (len(velocity), velocity.shape[1], velocity.shape[-1])
        reference = direction.detach()[:, :, None]  # [B,H,1,D]: only local K steps share d_h.
    else:
        assert direction.shape == (len(velocity), velocity.shape[-1])
        reference = direction.detach()[:, None, None]  # legacy global [B,1,1,D]
    return (velocity*reference).sum(-1)  # [B,H,K]


class DirectionalProgressLoss(nn.Module):
    def __init__(self, margin=.1, eps=1e-8):
        super().__init__()
        if margin < 0 or eps <= 0:
            raise ValueError("margin >= 0 and eps > 0 required")
        self.margin, self.eps = margin, eps

    def forward(self, velocities):
        return F.relu(self.margin-reference_progress(velocities, self.eps)).mean()


def objective(output, mode="directional", margin=.1, weight=1., eps=1e-8):
    if mode not in {"prediction_only", "nonzero_velocity", "directional"} or margin < 0 or weight < 0 or eps <= 0:
        raise ValueError("invalid ablation/loss settings")
    prediction = F.mse_loss(output["prediction"], output["target"])  # NO target detach
    velocity = output["velocities"]  # [B,K,D]
    if mode == "directional":
        progress = F.relu(margin-output_progress(output, eps)).mean()
    elif mode == "nonzero_velocity":
        progress = F.relu(margin-velocity.norm(dim=-1)).mean()
    else:
        progress = velocity.sum()*0
    return {"loss_total": prediction+weight*progress, "loss_pred": prediction, "loss_progress": progress}
