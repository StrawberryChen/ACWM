# Continuous Trajectory Dynamics：两个可训练方向版本

独立于原来的 single-step `continuous_dynamics` 配置，保留旧入口和权重。
第一阶段仍然没有 planner，不使用预训练、BN、SIGReg、teacher、重建、对比学习或物理监督。

## 两个实验

共同输入：当前 RGB `X_t`、未来候选/记录动作 `[A_t,...,A_{t+H-1}]`。
训练标签：真实未来帧 `[X_{t+1},...,X_{t+H}]` 的共享 Encoder latent。
**真实未来帧不进入动作编码、方向网络或 rollout。**

| 模块 | Action-only | State + Action |
|---|---|---|
| Action sequence `[B,H,2]` | 共享 MLP 2→128→192 + learned temporal positions + 2层/3头 Transformer + mean pooling | 相同 |
| 动作上下文 c | `[B,192]` | `[B,192]` |
| Direction Head 输入 | c `[B,192]` | concat(O_start,c) `[B,384]` |
| Direction Head | Linear→SiLU→Linear | 相同结构，但第一层输入更宽 |
| raw arrow q / normalized d | `[B,192]` | `[B,192]` |

两种配置除 `direction_mode`、W&B 名称和输出目录外一致。
由于 Direction Head 输入宽度不同，head 参数量不完全相同。
共同模块在 head 之前初始化，相同 seed 的初始视觉 Encoder 权重一致。
Action-only 的同一动作序列在不同起点具有相同箭头；State+Action 则可以不同。

## 确保方向网络被训练

\[
c, (u_0,...,u_{H-1})=ActionTrajectoryEncoder(\mathcal A)
\]

\[
q=DirectionHead(c)\quad\text{或}\quad q=DirectionHead([O_0,c]),\qquad
d=q/(\|q\|+\epsilon)
\]

不能仅把 detach 后的 d 放进 loss：那样独立 Direction Head 收不到梯度。
这里 **未 detach 的 d 也作为速度网络输入**：

\[
V^{h,k}=MLP([O^{h,k},u_h,c,d])\in\mathbb R^{192}
\]

输入拼接为 `[B,768]`，MLP 为 768→512→512→192，SiLU。
Action MLP 在所有 h 共享；动作序列不变，c 和 d 在整个 horizon 中计算一次、固定使用。
没有 episode time / frame index / integration index / FM time 输入。
动作顺序仅在 ActionTrajectoryEncoder 中通过相对序列位置编码表达。

预测 loss 与 progress loss 都可以沿 `V → d → DirectionHead → ActionTrajectoryEncoder`
反传。只有 progress 点积的 **reference 副本** d.detach() 被截断。
不增加新的方向监督/auxiliary loss。

## 真正的 open-loop trajectory

默认 H=16 个 **raw actions / raw transitions**，不是16个5-action打包步骤。
每个真实区间 K=4，dt=1/K；整个 horizon 共64次速度场计算。

\[
O^{0,0}=E(X_t),\quad O^{h,k+1}=O^{h,k}+V^{h,k}/K,
\quad O^{h+1,0}=O^{h,K}
\]

下一段从预测 endpoint 开始，不使用真实下一帧重置，不 teacher force。

| Tensor | shape |
|---|---|
| raw RGB window | `[B,H+1,3,96,96]` |
| actions | `[B,H,2]` |
| initial latent | `[B,192]` |
| true future labels / predicted endpoints | `[B,H,192]` |
| internal states | `[B,H,K+1,192]` |
| internal velocities | `[B,H,K,192]` |
| fixed horizon direction | `[B,192]` |
| projections | `[B,H,K]` |

共享 ViT/Linear Encoder 不变，真实未来 target 不 detach，两侧均反传。

\[
L_{pred}=\operatorname{mean}_{B,H,192}(\hat O_h-E(X_{t+h}))^2
\]

\[
L_{progress}=\operatorname{mean}_{B,H,K}\max(0,m-V^{h,k}\cdot\operatorname{sg}(d))
\]

\[
L=L_{pred}+\lambda L_{progress},\quad m=0.1,\;\lambda=1
\]

这里全局方向约束可能惩罚真实轨迹的折返；非零漂移折衷解仍可能存在。
这是待验证假设，不声称 trajectory/可训练箭头已经保证防住 collapse。

## Colab 命令

无需重做已有 raw-step NPZ。借用之前运行持久化保存的配置，只复用 data/device。
不要拿旧单步模型 checkpoint 恢复新模型，两种 direction_mode 的 checkpoint 也不能混用。

```python
%cd /content/ACWM

# 指向你之前单步实验保存的配置；若路径不同请相应修改。
DATA_CONFIG = "/content/drive/MyDrive/ACWM/continuous_dynamics/directional_v1/config.yaml"
```

实验 A：action sequence 定义箭头。

```python
!python -m experiments.continuous_dynamics.train \
  --config configs/continuous_trajectory_action.yaml \
  --data-config "{DATA_CONFIG}" \
  --output-dir /content/drive/MyDrive/ACWM/continuous_trajectory/action_only
```

实验 B：当前状态 + action sequence 定义箭头。

```python
!python -m experiments.continuous_dynamics.train \
  --config configs/continuous_trajectory_state_action.yaml \
  --data-config "{DATA_CONFIG}" \
  --output-dir /content/drive/MyDrive/ACWM/continuous_trajectory/state_action
```

默认各100 epochs、batch64、每5轮固定样本 validation 并保存可视化；两个输出目录独立。
直接编辑对应 YAML 的 `validation.interval` 改验证频率。最后一轮总会验证；
可视化只在已经验证且符合 `validation.visualization_interval` 的轮次（以及最后一轮）运行，
不会额外触发验证。训练日志和 `latest.pt` 每轮更新，`best_loss.pt` 只在验证轮比较。
H/K 增大意味着更多 frame encoder 和 dynamics 计算；显存不足时在 YAML 减小 data.batch_size，
两个对比实验保持一致，并且如果使用 --data-config，需要也同步其 batch_size（或用直接配置路径启动）。

验证、probe、续训沿用同一个入口，必须使用 **各自保存的 config.yaml**：

```python
RUN = "/content/drive/MyDrive/ACWM/continuous_trajectory/action_only"

!python -m experiments.continuous_dynamics.evaluate \
  --config "{RUN}/config.yaml" --checkpoint "{RUN}/checkpoints/latest.pt" --split val

!python -m experiments.continuous_dynamics.probe \
  --config "{RUN}/config.yaml" --checkpoint "{RUN}/checkpoints/latest.pt"

!python -m experiments.continuous_dynamics.train \
  --config "{RUN}/config.yaml" --checkpoint "{RUN}/checkpoints/latest.pt" --resume
```

## 诊断

保留原有 collapse、progress、真实相邻 latent 距离、PCA、完整 episode 轨迹和 probe。
内部积分图现在同时画整段 H×K 预测路径与 H 个真实 latent targets（共享同一 PCA basis）。

新增/调整：

- `train/direction_head_gradient_norm`、`train/action_trajectory_gradient_norm`：clip 前梯度范数。
- `val/pred_step_1 ... val/pred_step_H`：每个真实未来步的 prediction MSE。
- `val/identity_loss`：所有未来步都直接预测 O_start 的误差，对照“pred 很小是否只是表示收缩”。
- `direction/raw_norm`、`direction/feature_std`、`direction/velocity_cosine`。
- `direction/action_sequence_distance`：固定起点，随机动作序列产生的箭头差异。
- `action/velocity_distance` / cosine：同一起点，不同完整随机动作序列下初始速度差异。
- `action/shuffle_loss`：置换整个序列，**重新计算 context 和 direction 并完整 rollout**；不是只换第一个动作。
- `dynamics/velocity_cosine`：每个真实区间内部速度与其第一次速度的 cosine；
  与全 horizon 箭头的 cosine 单独记录在 `direction/velocity_cosine`。
- `dynamics/zero_reference_ratio`：raw direction q 接近0的比例。

这些诊断不参与 loss；direction head 可以有梯度但最终仍退化，需与动作 shuffle、latent std/rank 和 probe 联合判断。

## 测试

```bash
python -m unittest tests.test_continuous_dynamics tests.test_continuous_trajectory -v
```

测试两个 head 从 prediction 和 active progress loss 分别收到梯度、优化后权重改变、
未来帧不能改变预测或箭头、原始数据时间对齐、连续 open-loop、action顺序、
两个版本的状态输入差异、相同初始 Encoder、checkpoint 隔离、train/evaluate/probe/PNG smoke test。
旧单步回归测试同时运行；不代表真实任务效果验证。
