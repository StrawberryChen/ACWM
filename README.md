# Agent-Centric World Model (ACWM) v0.1

ACWM 将世界 latent 拆成可控的 **Agent State** 与任务环境对应的
**Environment State**。其结构约束是：

```text
Action -> Agent State -> Environment State
```

`EnvironmentTransition.forward(environment_state, next_agent_state)` 的接口中没有
action，因此动作无法绕过 Agent State 直接改变环境状态。

默认视觉 backbone 对齐 LeWorldModel：ViT-Tiny、patch size 14、hidden dim 192、
`pretrained: false`，端到端从头训练。Frame Encoder 使用最后一层 CLS token，
并接官方风格 projector：`Linear -> BatchNorm1d -> GELU -> Linear`，默认
hidden dim 为 2048、输出 dim 为 192。旧 CNN encoder 仍保留在 registry 中用于消融。

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
- `losses/`: agent、environment、goal consistency 和 SIGReg loss
- `trainer/`: 单步训练及多步 goal rollout 接口
- `planner/`: 可替换的 CEM latent planner
- `visualization/`: loss、相似度和 PCA embedding 调试图
- `configs/`, `utils/`: YAML 与组件 registry/factory

Agent 与 Environment embedding 分别应用 LeWorldModel SIGReg，通过随机投影与
Epps-Pulley 统计量约束 latent 接近各向同性高斯分布。默认使用 1024 projections、
17 knots 和 0.09 权重，并在 W&B 记录两个 latent 的标准差。

Planning rollout 内部继续遵守 `Action → Agent → Environment`，但 CEM cost 只接受
final/goal Environment State；Agent State 不进入任务目标评分。

v0.1 有意不包含图像重建、检测、scene graph、对象/关系标签或额外监督。
环境 latent 是联合表示，不宣称其内部已自动形成可解释的对象分解。

## Colab 训练、验证与 W&B

在 Colab 中安装 `.[colab]` 后执行 `wandb login`。配置中需要分别填写
`data.train_paths` 和 `data.val_paths`，避免轨迹级数据泄漏。每个 epoch 都会：

1. 训练；
2. 运行 one-step prediction validation；
3. 将 train/validation loss 写入 W&B 和 `outputs/metrics.jsonl`；
4. 保存可恢复训练的 latest checkpoint。

每 10 个 epoch（由 `validation.planning_interval` 控制）会在 LeWorld 对齐的
`swm/PushT-v1` 中闭环规划 50 episodes。Success rate 提升时保存
`outputs/acwm_best.pt`。验证 GIF 会上传 W&B；实际 Push-T 规划 MP4 只保存到
`outputs/videos`，不再同步到 W&B。

Colab 输出区会实时显示 training batch loss、prediction validation 平均 loss，
以及 planning episode 的累计 success rate；无需等待整个阶段结束才看到状态。
默认 planning 对齐 LeWorld Push-T eval：从 validation episode 采样 start state，
使用同一 episode 的 `+25` step state 作为 goal；CEM 使用 300 个 candidates、
30 次迭代、topk 30，规划 horizon 为 5，并每 5 个动作后重规划。这些值均可在 YAML 调整。

ACWM v0.1 没有 pixel decoder，因此 prediction GIF 展示真实连续帧与每一步
agent/environment latent prediction error；planning MP4 展示 CEM 在环境中的真实执行过程。
完整 Colab 操作清单见 [`COLAB.md`](COLAB.md)。LeWorld-aligned planning 需要 NPZ 内包含
完整 5D `states`，并且 action 会被转换为 `swm/PushT-v1` 使用的相对动作。
旧的 `lerobot/pusht_image` 转换数据只有 2D agent state，需改用
`scripts/prepare_pusht_zarr.py` 从原始 Diffusion Policy Push-T zarr 重新转换。

## 多步训练接口

`ACWMTrainer.compute_rollout` 接受常规 history 字段，并额外需要：

- `rollout_actions`: `[B,L,A]`
- `goal_frame`: `[B,C,H,W]`

它将预测得到的最终 Environment State 与编码后的 goal latent 比较。将
`training.mode` 设为 `rollout` 后，内置 Dataset 会按照 `rollout_length` 自动生成
这两个字段。

## 独立 Action-Contrastive 实验

新增单帧 CNN + action residual MLP 管线，唯一训练损失为 counterfactual action cosine ranking CE。
数据为独立的 raw-step train/val/test，不覆盖已有数据；collapse、shuffle、角度泛化、future retrieval 与 PCA 均单独记录。
完整 Colab 命令、输出路径及 planning 开关见 [ACTION_CONTRASTIVE.md](ACTION_CONTRASTIVE.md)。

第二个独立实验采用真实动作分支的条件高斯 NLL、行匹配与真实 latent SIGReg。
默认每 5 epoch 验证，100 epoch 结束后自动执行一次 test planning；原实验保持不变。
新数据准备和训练流程见 [ACTION_MATCHING_V2.md](ACTION_MATCHING_V2.md)。

这两条单 raw-step 管线的 planning 时间预算、CEM 参数与终点 MSE 成本已适配
LeWorld Push-T。模型仍然单步递推，不等同于官方五动作块 Predictor；旧 Colab
配置迁移及未对齐项目见 [LEWORLD_PLANNING_ALIGNMENT.md](LEWORLD_PLANNING_ALIGNMENT.md)。

## AVS-Target 独立原型

新增固定 motion target + 随机初始化 CNN CSE + 条件 Flow Matching 管线，
主实验只有 FM loss，不使用 pretrained/teacher/anti-collapse 正则，不运行 planning。
三种 target ablation、独立 LeWorld-style loss control、生成评估和 frozen probes
的说明与 Colab 命令见 [AVS_TARGET.md](AVS_TARGET.md)。

AVS-only 保持不变，新增固定 terminal measurement 的 Terminal-only / AVS+Terminal。
两分支分别 mean 后加权 FM，prediction/probe 审核后才显式运行 terminal-space CEM/MPC。
配置、诊断流程及命令见 [AVS_TERMINAL.md](AVS_TERMINAL.md)。
