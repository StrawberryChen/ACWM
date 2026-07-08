import torch

from losses import SIGRegLoss


def test_sigreg_is_differentiable():
    embeddings = torch.randn(2, 16, 8, requires_grad=True)
    loss = SIGRegLoss(knots=9, num_projections=32)(embeddings)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
