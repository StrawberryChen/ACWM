# Sequential velocity + SIGReg

Independent experiment `velocity_sigreg`. The arrow/progress variants are unchanged.
Use the existing episode-disjoint raw-step NPZ data: 26 frames and 25 aligned
2-D actions per training window. No simulator state goes into training.

For each raw step, encode `z_j = E(O_j)`, take `v* = z_{j+1} - z_j` (one raw
step is the time unit) and feed `z_j + tau * v*` and `a_j` to a shared MLP.
Half the examples use `tau=0` by default; the rest sample a phase inside the
straight bridge. No separate original-state input, Gaussian noise, progress
loss, direction head, or Euler rollout in the training objective. Gradients
reach the shared encoder; the velocity MSE supervises the field directly.
SIGReg only sees real frame latents `[26,B,192]`; weight 0.09, 17 knots,
256 projections by default (compute trade-off, not an official replication).

At inference the state starts at the current frame embedding. Integrate with
the current action held fixed across two Euler half-steps, switch to the
next action, and repeat. Readouts at raw steps 5/10/15/20/25 give
`[B,5,192]`; CEM retains internal raw endpoints for per-executed-step
diagnostics. Goal frames are +25 raw steps by default. CEM settings, relative
action conversion, and real-environment success are otherwise unchanged;
MPC executes one raw action before replanning, not the full 25-action block.

## Colab

Run the [notebook](notebooks/ACWM_Trajectory_Colab.ipynb) from 01 to 06.
Cell 02 defaults to `velocity_sigreg`, 20 epochs, batch 32, validation
every 5 epochs, and 10 planning cases. First epoch pauses for Drive
verification; then continue from that checkpoint. No data re-download
and no manual checkpoint paths. Saved runs are under
`/content/drive/MyDrive/ACWM/trajectory_runs/velocity_sigreg/<run_name>`.

CLI equivalents from the repository root (after mounting Drive and installing
`.[colab-trajectory]`):

```bash
python -u -m experiments.continuous_dynamics.train --workflow configs/colab_velocity_sigreg.yaml --stop-after-epoch 1
```

Then in Colab, after confirming the checkpoint in Drive:

```python
from experiments.continuous_dynamics.workflow import Workflow
flow = Workflow.from_yaml("configs/colab_velocity_sigreg.yaml")
flow.train_remaining(drive_confirmed=True)
flow.probe()
flow.plan()
```

## Reading results

- `val/velocity_start_mse` versus `val/velocity_zero_mse` asks whether the
  learned field beats predicting zero velocity at the real start of each raw step.
- `action/velocity_shuffle_loss_delta > 0` asks if the correct action helps
  local velocity regression; `action/shuffle_loss_delta > 0` asks if correct
  actions help the open-loop rollout. Neither alone is a causal intervention.
- `val/loss_pred` is evaluation-only MSE on the five readouts; compare it to
  `val/identity_loss`, then inspect `val/pred_step_{5,10,15,20,25}` for drift.
- `collapse/feature_std`, `collapse/effective_rank`, and physical Probe are
  useful alongside these errors. Keep model/data/seed fixed when ablating SIGReg.
