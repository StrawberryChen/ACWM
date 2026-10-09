# Action-Contrastive Latent World Model

独立版本 `action_contrastive`，入口为 `train_action_contrastive.py`。现有版本及数据目录保持原样。

## 数据与模型

从原始 Diffusion Policy Push-T zarr 按 episode 分为 train/val/test（80/10/10），不跨 episode 取样。
`image[t] -> image[t+1]` 使用同索引的 `action[t]`，一个模型步就是一个 raw environment step。
原始绝对目标动作转为 `clip((target_xy-agent_xy)/100, -1, 1)`，与 swm/PushT-v1 的 relative action 接口一致。
裁剪会丢弃超过 100 像素的动作幅值，这是本版按指定 bounded-action 实验采用的预处理；不是声称原始绝对动作被无损转换。
Simulator state 仅留在文件中供环境 reset 使用；训练 batch 不包含 state。

数据流：

```
image_t / image_next [B,3,96,96]
  -> shared CNN (32,64,128,256 channels, stride=2, SiLU)
  -> global average pooling -> Linear(256,D)
  -> raw O_t / O_next [B,D], default D=128

A_true [B,2] -> candidates [B,1+K,2], default [A,-A]
  -> shared action MLP 2->32->32
concat(O_t, A_emb) [B,160] -> MLP 160->256->256->128
O_pred = O_t + delta [B,128]
cosine(O_pred_candidates, O_next) / temperature -> CE(label=0)
```

唯一训练损失是 CE；两个 encoder 分支都接收梯度。不使用 latent MSE、KL、SIGReg、variance/covariance loss 或重建。
内部状态不做 L2 normalization，仅在打分时 normalize。时序 rollout 必须沿时间递推；候选序列批量并行。

## Colab

启用 GPU，然后运行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/StrawberryChen/ACWM.git
%cd /content/ACWM
!git pull --ff-only
!pip install -e '.[colab,dev]'
!wandb login
!python scripts/prepare_action_contrastive.py \
  --output-root /content/drive/MyDrive/ACWM \
  --config-output configs/colab_action_contrastive.yaml
!python train_action_contrastive.py --config configs/colab_action_contrastive.yaml
```

如果 `/content/ACWM` 已存在，跳过 clone，直接 cd 后 pull。需要重新准备本版本 raw-step 数据；旧版本数据不删除、不覆盖。
每 5 epoch 做一次 validation，可在训练命令追加 `--validation-interval 5`，或设置 YAML `validation.interval: 5`。
包括 validation loss、shuffle、角度泛化、retrieval、representation diagnostics 与图；最后一轮也执行 validation。
每轮仍训练、记录训练指标及保存 latest 权重。planning 由独立的 `planning_enabled/planning_interval` 控制。
准备脚本复用已有下载压缩包，数据存到 `ACWM/action_contrastive/data/{train,val,test}`。
相同 manifest 可重跑；更换 split seed/比例/动作尺度时使用新的 output-root，防止混入旧 split。

验证和最终独立测试：

```bash
!python train_action_contrastive.py --config configs/colab_action_contrastive.yaml \
  --mode evaluate --split val \
  --checkpoint /content/drive/MyDrive/ACWM/action_contrastive/outputs/checkpoints/best_ranking.pt
!python train_action_contrastive.py --config configs/colab_action_contrastive.yaml \
  --mode evaluate --split test \
  --checkpoint /content/drive/MyDrive/ACWM/action_contrastive/outputs/checkpoints/best_ranking.pt
```

只有确认表征/转移有效后再运行 planning：

```bash
!python train_action_contrastive.py --config configs/colab_action_contrastive.yaml \
  --mode plan --split test \
  --checkpoint /content/drive/MyDrive/ACWM/action_contrastive/outputs/checkpoints/best_ranking.pt
```

若希望训练中每 20 epoch planning，将生成 YAML 中 `validation.planning_enabled` 设为 `true`。
规划 horizon=16 raw steps，MPC 默认执行 1 步后重规划，候选动作限制在 [-1,1]，代价为 `1-cosine(O_final,O_goal)`。
复用当前 `CEMPlanner` 与 `PlanningEvaluator`。dataset goal 默认是 +25 raw steps 的环境状态，success 是 swm 环境定义的状态目标成功率（0–100%）；不等同于标准 gym-pusht 固定 T 目标区域 coverage。
swm 环境输出负状态距离，本版记录 mean best/final goal distance 与 reward，不把它们误标为 coverage。
另记录规划耗时、动作幅度/跳变、latent imagined transitions/sec。

恢复训练：

```bash
!python train_action_contrastive.py --config configs/colab_action_contrastive.yaml --resume \
  --checkpoint /content/drive/MyDrive/ACWM/action_contrastive/outputs/checkpoints/latest.pt
```

## 输出与消融

`.../action_contrastive/outputs/` 包含配置、`logs/metrics.jsonl`、`plots/*.png`、`videos/`、`checkpoints/`。
保存 `latest.pt`、`best_ranking.pt`；开启训练内 planning 后保存 `best_planning.pt`。
checkpoint 包含完整模型、三个子模块、optimizer、scheduler、epoch、config、metrics 和 RNG state。
数值指标上传 W&B；planning 视频只保存磁盘。九张实验曲线以及 PCA（timestep、episode、action direction）每个 epoch 更新。
不额外推断或伪造 contact/reward 标签；原始 zarr 没有这些分析标签时不绘制对应着色。

YAML 支持 `model.latent_dim: 32/64/128/256`、`loss.temperature`、`loss.num_negatives`、
`loss.negative_type: opposite/random/orthogonal/mixed`、`model.architecture: residual_mlp/concat_mlp/film`。
多个 opposite negatives 时，第一项严格为 `-A`，其余角度围绕反向扩展。多个 orthogonal negatives 交替方向并调整幅值以避免重复。

指标需要共同判断：val ranking/gap、feature std、effective rank、pairwise cosine、O/A shuffle drop、angle generalization、retrieval horizon 曲线。
effective rank 使用 centered feature 的奇异值熵；retrieval 的 N=min(32,可用样本数)，实际 N 和样本数单独记录。
角度探针仅评估动作范数 <=1 的样本，使旋转后动作保持原幅值、角度且不越过环境动作范围；每个角度的实际样本数也记录。
retrieval distractor 来自同一个 held-out split 的不同 transition，按行无放回采样。评估固定 seed、相同起点与 candidate bank 跨 horizon 比较。
重复或视觉近似状态会使 retrieval 更难；该任务不能单独证明 rollout 物理正确。
ties 不算 ranking/retrieval 成功，防止 collapsed representation 通过 argmax tie 假得高分。
零动作下 A=-A，无可辨别标签；默认保留并记录比例，可以通过 `data.min_action_norm` 做显式消融。

测试：`python -m pytest tests/test_action_contrastive.py -q`。

真实环境集成测试：`SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ACWM_TEST_PUSHT=1 python -m pytest tests/test_action_contrastive.py -q`。
本次实现已通过全部 10 项测试，包括训练/恢复/独立测试评估、共享 encoder 梯度、时间对齐、数据 split、CEM、真实环境执行和 MP4 保存。
这是接口与数值流程测试，尚未完成真实数据训练，不代表 ranking、representation 或 planning 效果已经验证。
额外运行的旧版 dataset/loss/training/planner 回归中有 5 项通过、2 项失败：旧测试要求完全固定的 metrics keys，
但旧 Trainer 已额外记录 `learning_rate`、`global_step`、`gradient_norm`。本次没有修改这些旧实现或测试。
