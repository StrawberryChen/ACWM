"""Linear conditional flow matching and explicit Euler; no auxiliary objective."""
import torch
from torch.nn import functional as F


def flow_training_pair(target, noise, time):
    assert target.shape == noise.shape and time.shape == (len(target),)
    assert not target.requires_grad
    s = time[:, None, None]
    return (1-s)*noise+s*target, target-noise


def flow_loss(predicted, target_velocity):
    assert predicted.shape == target_velocity.shape and not target_velocity.requires_grad
    return F.mse_loss(predicted, target_velocity)  # the ONLY training loss


@torch.no_grad()
def euler_sample(predictor, state, actions, noise, steps=16):
    if steps < 1:
        raise ValueError("Euler steps must be positive")
    value = noise.clone()  # no access to real future frames/target
    for k in range(steps):
        time = value.new_full((len(value),), k/steps)
        value = value+predictor(value, time, state, actions)/steps
    return value
