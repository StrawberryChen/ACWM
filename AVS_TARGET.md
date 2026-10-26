# AVS-Target End-to-End World Model v1

独立管线：`experiments/avs_target/`。不覆盖已有模型或未提交的 planning 改动。
本文描述保留的 AVS-only；新增 Terminal-only / AVS+Terminal 及独立 terminal-space
planning 见 [AVS_TERMINAL.md](AVS_TERMINAL.md)，不改变本实验。
研究问题是：固定 motion-sensitive future measurement 是否足以诱导有用的当前状态表示？
这是一项待验证假设，不是已经证明的 anti-collapse 定理。

## 数据流与 shapes

| 环节 | 默认 shape |
|---|---|
| 动态 sliding window 原始帧 | B×17×3×96×96 |
| 当前帧 / 动作序列 | B×3×96×96 / B×16×2 |
| 随机初始化 CNN CSE | B×128 |
| 固定 AVS target | B×16×8×6×6 → B×16×288 |
| noise / X_s / target velocity | B×16×288 |
| 动作编码 | B×16×2 → 64 → 128 → 256 |
| motion projection | B×16×288 → B×16×256 |
| Transformer 输入 | B×17×256（当前状态 token + 16 future tokens） |
| 输出 velocity | B×16×288 |

样本严格使用 X_t...X_t+H 和 A_t...A_t+H-1，第零个测量是 AVS(X_t,X_t+1)。
一个 world step 就是一个 raw action；不复制窗口图片、不使用未来 Encoder。
训练 batch 不含 simulator state；NPZ 的 states 仅供独立 probe 读取。

CSE 是 32/64/128/256 四层 stride-2 CNN + SiLU + GAP + Linear(256,128)。
不做 L2/LayerNorm/BatchNorm，不加载 pretrained 权重。
Flow Predictor 是 4-layer、8-head、256D、MLP ratio=4 的双向 Transformer。
future token=motion projection+action embedding+world position+flow time embedding；
当前状态作为单独 context token。world time h 和生成时间 s 使用两个不同的 sinusoidal 编码。
Transformer 内部标准 LayerNorm 不等于给 CSE 输出施加 normalization/anti-collapse。

## 固定 AVS-style 算子

这是 learning-free detector，不声称复现某篇 AVS 论文，也不是 neural optical flow。
RGB→灰度→可选固定 Gaussian smoothing；temporal difference 拆分 ON/OFF，作为变化 gating。
再用固定局部 ON/OFF 空间对比响应，构造前帧向后帧的延迟方向相关，减去反向相关、整流，
对 [1,2,4] pixel shifts 平均，再固定 average-pool。亮/暗物体同向移动应激活一致方向。
边界零填充，不会用 roll 从图像另一侧绕回。

8 方向按图像坐标顺时针：右、右下、下、左下、左、左上、上、右上。
支持 4/8 方向、3/6/12 分辨率。亚像素运动、遮挡、重复纹理和相机变化可能造成弱/错误响应，
必须先观察 target 质量；direction channel 不是 simulator motion ground truth。
算子无 Parameter，计算全程 no_grad；不接受 Flow loss 的梯度。

每通道 mean/std 只拟合 train 的 UNIQUE adjacent transitions，不对重叠窗口重复计数。
std 下限 1e-6。`avs_stats.json` 保存算子 fingerprint、来源路径/大小/mtime、mean/std。
验证/生成/恢复复用同一统计量。改 target、分辨率、方向数或训练数据时使用新 output-dir。

## 唯一训练目标

```
epsilon ~ N(0,I), s ~ Uniform(0,1)
X_s = (1-s)*epsilon+s*Y
V_target = Y-epsilon
L_total = mean((Flow(X_s,s,CSE(X_t),A)-V_target)**2)
```

梯度只经过 Flow、Action Encoder 和 CSE。没有 SIGReg/VICReg、KL、contrastive、
重建、EMA、teacher、可训练 target projection 或 goal latent loss。
默认 optimizer weight_decay=0，避免额外参数收缩。
生成时固定 O/A，从 noise 开始 Euler 更新 `X <- X + v(X,s,O,A)/K`，不接触真实未来。

固定 Y 不保证 CSE 被使用：模型仍可能利用动作、X_s 中的 target 信息，或学习条件不足的边缘分布。
因此除整体 FM loss 外，还测 s=0/0.1/0.5/0.9 下的条件依赖性，以及完整生成时的 shuffle 退化。

## Ablations 与 baseline

- AVS：8×6×6=288。
- Pixel difference：RGB 相邻帧差固定 average-pool 到 6×6，D=108，保留正负号。
- Fixed random projection：同一 108D RGB difference，固定 seed Gaussian projection 到 288D。
- 独立 LeWorld-style control：同一随机 CNN、共享真实帧编码、确定性 action MLP、MSE+原工程 SIGReg。

最后一项是控制 backbone 的 LeWorld-style loss 对照，不是官方 ViT/AdaLN 的完全复现。
它使用独立 baseline.py；主 Flow trainer 拒绝 baseline 配置，不会偷偷加入正则。
三个 target 的维数/统计尺度不同，不可单凭 target-space MSE 横向宣布哪个表征最好；
需要比较相同数据 split 上的 probes、依赖性和生成结果。

## 验证与保存

默认训练 100 epoch，每 5 epoch validation，最后一轮也验证；不运行 CEM/planning。
W&B 记录 train/val flow_loss、真实 CSE 的 feature_std、covariance effective_rank、
pairwise_cosine、state_variance、O/A shuffle delta、target energy/variance/zero ratio/channel distribution，
以及 generation MSE/cosine/magnitude error。AVS 另报 direction accuracy。
记录 CSE 和 Action Encoder 的梯度范数。effective rank 使用中心化 covariance eigenvalues 的熵，
不能代替绝对尺度指标；须与 feature_std 一起看。

验证使用固定采样与 noise/time。生成覆盖 K=4/8/16/32，H=1/2/4/8/16（不超过训练 H）。
各 H 独立截取动作再生成，不只是截取长预测；不同 K 复用初始 noise。
默认 generation_samples=64。方向准确率仅统计非零 GT motion pixel，静止时省略 accuracy，
报告 valid pixels=0；Pixel/RP 不伪造 direction accuracy。

首次保存 50 个随机 validation windows 的 target 图：真实两帧、8 通道、argmax 与 magnitude。
样本不足时明确显示实际数量。每次 validation 保存生成轨迹 MP4：
左=真实未来帧，中=真实 measurement magnitude，右=生成 measurement magnitude。
不是像素重建，也不是 agent planning。PNG 同步 W&B，MP4 只保存 Drive。

每 epoch 保存 latest.pt；按 validation 最大 K、完整 H 的 generation MSE 选 best_generation.pt。
包含 optimizer/scheduler/RNG、固定 target buffers、原始 random CSE，支持 resume。
test 不参与模型选择。对照保存 best_prediction.pt。

## Colab

启用 GPU 并挂载 Drive，代码同步后运行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/ACWM
!pip install -e '.[colab]'
!wandb login
!python scripts/prepare_avs_target.py \
  --output-root /content/drive/MyDrive/ACWM \
  --config-output configs/colab_avs_flow.yaml
!python -m experiments.avs_target.train --config configs/colab_avs_flow.yaml
```

数据单独存入 ACWM/avs_target/data/{train,val,test}，默认 (absolute_target-agent_xy)/512，
按 workspace scale 归一化、不 clipping。相同 manifest 可重跑，不删除旧数据。
可以复用旧 raw-step NPZ，但 action_scale 必须保留真实元数据（原来是 100 就不能填 512）；
不能使用 5-action-block NPZ。当前脚本另存 episode，不存预计算 sliding windows。

输出：`/content/drive/MyDrive/ACWM/avs_target/outputs/avs/`。

同数据 ablation（无需重新准备数据，各自使用独立输出目录）：

```python
!python -m experiments.avs_target.train --config configs/colab_avs_flow.yaml --target pixel_diff
!python -m experiments.avs_target.train --config configs/colab_avs_flow.yaml --target random_projection
!python -m experiments.avs_target.baseline \
  --config configs/avs_leworld_style.yaml --data-config configs/colab_avs_flow.yaml
```

恢复与独立测试：

```python
!python -m experiments.avs_target.train --config configs/colab_avs_flow.yaml \
  --checkpoint /content/drive/MyDrive/ACWM/avs_target/outputs/avs/checkpoints/latest.pt --resume
!python -m experiments.avs_target.train --config configs/colab_avs_flow.yaml \
  --mode evaluate --split test \
  --checkpoint /content/drive/MyDrive/ACWM/avs_target/outputs/avs/checkpoints/best_generation.pt
```

最后冻结 CSE 做 evaluation-only probes：

```python
!python -m experiments.avs_target.probe --config configs/colab_avs_flow.yaml \
  --checkpoint /content/drive/MyDrive/ACWM/avs_target/outputs/avs/checkpoints/best_generation.pt
```

linear ridge 与 2-layer MLP 只训练 probe heads；CSE frozen，不返回任何 probe 梯度。
位置报告像素 RMSE，方向用 sin/cos 回归再计算 circular angular MAE。
统计量只用 probe train 拟合，MLP 用 val 选权重，最后一次 test。
同时评估原始随机 CSE 和 constant mean-state 控制；输出写入对应 checkpoint 实验的 probes/。

修改 H 时 validation.horizons 需要含该 H；flow_steps 可直接配置。
每个 worker 默认缓存 8 个 uint8 episodes；随机 NPZ 解压可能慢，可按 RAM 调大 cache_episodes，
或 num_workers=0 避免 worker 重复缓存。cache_episodes=0 表示不限制缓存。

不要直接套用旧 latent goal cost：本模型输出 motion trajectory，不是未来 state latent。
先验证 target、条件依赖性、生成结果和 probes；失败时不自动增加 auxiliary loss。
