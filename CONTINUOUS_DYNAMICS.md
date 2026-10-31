# Action-Conditioned Continuous Latent Dynamics v1

这是独立实验，不替换 ACWM、LeWorld、Motion Token、Forward/Inverse、AVS 或 matching 管线。
代码在 `experiments/continuous_dynamics/`。第一阶段没有 planning、success rate 或规划视频。

## 数据与模型

复用已有 **raw-step、episode-disjoint** NPZ 和图像预处理。每个样本严格为
`frames[t:t+2]`、`actions[t:t+1]`，不是 5-action 打包，不是多帧 history。
当前 NPZ 的动作是 `(absolute_target_position - agent_position) / action_scale`，默认 scale=512；
不是绝对 target position。预处理时已有的转换不是新的监督项；训练只读图像与已保存的动作。
不能把旧的 packed-action 数据直接传进来。Dataset 不向模型 minibatch 返回 simulator state。
states 仅由独立 probe 和 read-only 可视化读取，绝不进入 world-model loss。

| 阶段 | shape |
|---|---|
| 前后两帧 RGB，float [0,1] | `[B,2,3,96,96]` |
| 当前 raw action | `[B,1,2]` |
| 同一随机初始化 ViT-Tiny，两帧共享参数 | `[2B,192]` |
| 最后一层 CLS + Linear(192,192) | `O_start,O_target: [B,192]` |
| ActionEncoder: 2→128→192，SiLU | `[B,192]` |
| VelocityField: concat→384→512→512→192，SiLU | `[B,192]` |
| K=4 次 Euler，固定同一个 A | states `[B,5,192]`, velocities `[B,4,192]` |
| 最终预测 | `[B,192]` |

ViT 为 12 layers / 3 heads / patch16 / 输入96，保留 ViT 自身 LayerNorm，
不使用 BN、预训练、teacher、EMA、target detach、SIGReg、方差/协方差约束、
contrastive、重建、AVS/HOG target、物理监督、Gaussian noise 或 Flow Matching。
VelocityField 只接收 O 和 A，没有 episode/frame time、integration index 或 FM time。

## 精确目标函数与理论边界

对每个样本独立计算：

\[
O^0=E(X_t),\quad V^k=V_\theta(O^k,A_t),\quad O^{k+1}=O^k+V^k/K
\]

\[
d=\operatorname{sg}\left(\frac{V^0}{\|V^0\|_2+10^{-8}}\right),\quad
p_k=\langle V^k,d\rangle
\]

\[
L_{pred}=\operatorname{mean}_{B,D}(O^K-E(X_{t+1}))^2,
\quad L_{progress}=\operatorname{mean}_{B,K}\max(0,m-p_k)
\]

\[
L=L_{pred}+\lambda L_{progress},\quad m=0.1,\;\lambda=1
\]

两侧 Encoder 和全部 K 步都正常反传，只有 d detach。无额外 loss。
这不是“已经证明不会坍缩”的方法：

- 若所有 progress ≥m>0，则 `(O_pred-O_start)·d ≥m`，因此常数 Encoder 不能同时令两项为零。
- 但 **V=0 时 d=0，progress loss=m，却有零梯度**。严格按技术路线实现，不加随机方向或 norm fallback 掩盖它。
- 即使 V 非零，常数 Encoder + 固定单向速度仍可形成低损失折衷。
  因 MSE 按192维平均，若位移约m，则 prediction loss 仅约 `m²/192=0.0000521`。
  所以总 loss 很小不等于表示未坍缩，也不证明学到 block 或物理状态。
- margin=0 允许完全静止；K=1 没有后续中间路径约束。
- Euler 是有限步离散近似；内部曲线没有真实 intermediate GT，PCA 外观不是连续物理路径的证明。

## 默认训练与记录

100 epochs，batch64，AdamW lr=3e-4，constant LR，weight_decay=0，gradient clip=1。
**每个 epoch validation**（默认固定采样1024个 raw transitions；不是完整验证集 loss）。
第1轮、每5轮、最后一轮保存 PCA/轨迹图，其间每轮仍记录所有诊断。
W&B 在线记录标量与静态 PNG，训练输出有 batch 进度条和每轮摘要。

- `train/loss_total`, `train/loss_pred`, `train/loss_progress`；对应 `val/*`。
- `collapse/feature_std`：按 observations 统计各维 std 的均值。
- `collapse/effective_rank`：中心化 covariance 特征值的 entropy effective rank；全常数设为0。
- `collapse/pairwise_cosine`、`collapse/zero_latent_ratio`、`collapse/state_variance`。
- `dynamics/latent_transition_distance/{mean,std,min,max}`，真实相邻帧 latent 距离。
- `dynamics/velocity_norm/{mean,std,min,max}`，所有 K 步速度。
- `dynamics/progress/{mean,std,min,max}`、`dynamics/progress_violation_ratio`。
- `dynamics/velocity_cosine`，仅诊断，不加入 loss；`dynamics/zero_reference_ratio`。
- `action/velocity_distance`, `action/velocity_cosine`：同一 O，8个 uniform[-1,1]² A 的两两速度比较。
  这是敏感性测试，不是这些随机动作有正确未来 GT 的反事实评测。
- `action/normal_loss`, `action/shuffle_loss`, `action/shuffle_loss_delta`。
  shuffle 在整个 validation 样本上构造无固定点索引置换；动作值本身可能重复。
- 物理状态着色的 validation PCA、完整 validation episode 轨迹、一个 raw step 的内部 Euler 轨迹与真实 target。
  所有轨迹使用同一 validation-fitted PCA basis，并保存原始高维向量 NPZ。

检查 feature std / rank / transition distance 与 probe，不要用 loss 最小或 cosine 单一指标断言成功。

## Colab

进入仓库并安装依赖、挂载 Drive、登录 W&B 后执行：

```bash
python -m experiments.continuous_dynamics.train \
  --config configs/continuous_dynamics.yaml \
  --data-config configs/colab_avs_flow.yaml
```

`--data-config` 只借用已准备好的 raw-step NPZ 路径、动作单位与 device 等数据设置，
不继承旧 AVS 模型/target/loss。新输出默认在旧 output_dir 的同级目录：
`/content/drive/MyDrive/ACWM/avs_target/outputs/continuous_dynamics/directional/`。
可用 `--output-dir /content/drive/MyDrive/ACWM/continuous_dynamics/run_01` 明确指定。
若直接编辑基础 YAML 的 data 路径，则不需要 `--data-config`。

**已有 AVS raw-step NPZ 不需要重新准备。** 若没有该数据，复用现有提取脚本：

```bash
python scripts/prepare_avs_target.py --output-root /content/drive/MyDrive/ACWM
```

该命令只提取 raw RGB/action/state NPZ 并生成 `configs/colab_avs_flow.yaml`；
本实验不训练或使用 AVS target。不要删除旧数据或其他实验输出。

保存：`checkpoints/latest.pt`、`checkpoints/best_loss.pt`（按 validation total loss，**不是 best representation**）、
`config.yaml`、`metrics.jsonl`、`diagnostics.pt`、`visualizations/*.png` 和 `latents_*.npz`。
Checkpoint 包含初始随机 encoder、optimizer、epoch、RNG、数据内容哈希。
已有 latest 的目录默认拒绝新训练，避免意外覆盖。新消融请指定新 output-dir。

独立验证与 test：

```bash
RUN=/content/drive/MyDrive/ACWM/avs_target/outputs/continuous_dynamics/directional
python -m experiments.continuous_dynamics.evaluate \
  --config "$RUN/config.yaml" --checkpoint "$RUN/checkpoints/latest.pt" --split val
python -m experiments.continuous_dynamics.evaluate \
  --config "$RUN/config.yaml" --checkpoint "$RUN/checkpoints/latest.pt" --split test
```

冻结 encoder 的 probe（训练独立 Linear ridge 和小 MLP，test 不参与拟合/选模）：

```bash
python -m experiments.continuous_dynamics.probe \
  --config "$RUN/config.yaml" --checkpoint "$RUN/checkpoints/latest.pt"
```

Probe 对比 trained encoder、相同初始 random encoder、constant predictor。
position 误差单位为 pixels，angle 用 sin/cos 回归后转换为角度 MAE，避免周期边界问题。
World model checkpoint 不被 probe 更新。只在独立 probe 中使用物理标签。

续训（保持原配置；这里只恢复中断运行，不隐式延长训练）：

```bash
python -m experiments.continuous_dynamics.train \
  --config "$RUN/config.yaml" --checkpoint "$RUN/checkpoints/latest.pt" --resume
```

## 消融

三个独立 YAML：

| 配置 | Loss |
|---|---|
| `configs/continuous_prediction_only.yaml` | Prediction only |
| `configs/continuous_nonzero_velocity.yaml` | Prediction + mean ReLU(m-‖Vᵏ‖) |
| `configs/continuous_dynamics.yaml` | Prediction + Directional Progress |

其余步骤相同，将训练命令 `--config` 替换即可。K、margin 可在 YAML 修改，或命令行覆盖：

```bash
python -m experiments.continuous_dynamics.train \
  --config configs/continuous_dynamics.yaml --data-config configs/colab_avs_flow.yaml \
  --integration-steps 8 --margin 0.05 \
  --output-dir /content/drive/MyDrive/ACWM/continuous_dynamics/K8_m005
```

分别测试 K=1/2/4/8、margin=0/.01/.05/.10/.20，并使用不同目录。
所有生效覆盖均写入保存的 config.yaml 和 W&B config。
第一版不提供 planning 命令；未来可复用 `model.rollout(O, actions)`，每个 raw A 都重新执行 K 次积分。

## 测试

```bash
python -m unittest tests.test_continuous_dynamics -v
# 或 pytest -q tests/test_continuous_dynamics.py
```

覆盖真实 ViT shape、无BN、前后帧梯度、ActionEncoder/全部积分步梯度、逐样本方向与detach、
Euler K=1/2/4/8、三种 loss、零速度 stationary point、常数 Encoder 低loss折衷、
raw-step 对齐、episode split 泄漏检查、只读 diagnostics，及 train/checkpoint/evaluate/resume/probe/PNG 端到端 smoke test。
Smoke test 只证明代码可运行，不是非坍缩或真实任务性能证据。
