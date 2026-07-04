# Agent-Centric World Model (ACWM) v0.1

ACWM 将世界 latent 拆成可控的 **Agent State** 与完整观察对应的
**Environment State**。其结构约束是：

```text
Action -> Agent State -> Environment State
```

`EnvironmentTransition.forward(environment_state, next_agent_state)` 的接口中没有
action，因此动作无法绕过 Agent State 直接改变环境状态。

## 数据约定

每个轨迹为一个 `.npz` 文件：

- `frames`: `[T,C,H,W]` 或 `[T,H,W,C]`，uint8 或 `[0,1]` float
- `actions`: `[T-1,A]`，第 `t` 项表示 `frame[t] -> frame[t+1]`

这层格式是 Push-T 原始数据与模型之间的薄适配层。若原数据来自 zarr/replay
buffer，只需转换为上述轨迹，或实现返回相同字段的新 Dataset，无需改模型。

## 使用

```bash
python -m pip install -e '.[dev,colab]'
pytest -q
python train.py --config configs/default.yaml
```

训练前在 YAML 的 `data.train_paths` 和 `data.val_paths` 填入轨迹文件。模型、优化器、学习率、latent
维度、rollout 长度及 loss 权重均由配置控制。

在 Colab 中无需手动寻找数据：`scripts/prepare_pusht.py` 会从 Hugging Face 的
`lerobot/pusht_image` 下载数据，按完整 episode 划分 train/validation，转换为
ACWM NPZ，并生成训练配置。详见 [`COLAB.md`](COLAB.md)。

## 模块边界

- `dataset/`: 通用轨迹切窗
- `models/encoder/`: 独立 Agent / Environment 编码器
- `models/predictor/`: 两条严格分离的状态转移路径
- `losses/`: agent、environment、goal consistency latent loss
- `trainer/`: 单步训练及多步 goal rollout 接口
- `planner/`: 可替换的 CEM latent planner
- `visualization/`: loss、相似度和 PCA embedding 调试图
- `configs/`, `utils/`: YAML 与组件 registry/factory

v0.1 有意不包含图像重建、检测、scene graph、对象/关系标签或额外监督。
环境 latent 是联合表示，不宣称其内部已自动形成可解释的对象分解。

## Colab 训练、验证与 W&B

在 Colab 中安装 `.[colab]` 后执行 `wandb login`。配置中需要分别填写
`data.train_paths` 和 `data.val_paths`，避免轨迹级数据泄漏。每个 epoch 都会：

1. 训练；
2. 运行 one-step prediction validation；
3. 将 train/validation loss 写入 W&B 和 `outputs/metrics.jsonl`；
4. 保存可恢复训练的 latest checkpoint。

每 5 个 epoch（由 `validation.planning_interval` 控制）会在官方
`gym_pusht/PushT-v0` 中闭环规划 100 episodes。Success rate 提升时保存
`outputs/acwm_best.pt`。验证 GIF 与实际 Push-T 规划 MP4 会同时保存到
`outputs/videos` 并上传 W&B。

ACWM v0.1 没有 pixel decoder，因此 prediction GIF 展示真实连续帧与每一步
agent/environment latent prediction error；planning MP4 展示 CEM 在环境中的真实执行过程。
完整 Colab 操作清单见 [`COLAB.md`](COLAB.md)。若你有精确的 goal RGB 图，可设置
`environment.goal_image`；否则评估器用 `goal_reset_state` 在 Push-T 中渲染目标图。

## 多步训练接口

`ACWMTrainer.compute_rollout` 接受常规 history 字段，并额外需要：

- `rollout_actions`: `[B,L,A]`
- `goal_frame`: `[B,C,H,W]`

它将预测得到的最终 Environment State 与编码后的 goal latent 比较。将
`training.mode` 设为 `rollout` 后，内置 Dataset 会按照 `rollout_length` 自动生成
这两个字段。
