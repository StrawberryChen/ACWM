# Action Matching V2

独立第二版：保留第一版及其他所有模型。入口为 `train_action_matching.py`，
配置为 `configs/action_matching_v2.yaml`，不是原来的 `train.py`。

## 数据和模型

每个样本从相同完整物理起点执行 K 个不同动作，记录各自真实下一帧。
默认 K=2：数据集动作与其反方向；每隔 5 raw step 采集一次分支。
这里 stride=5 仅是采样间隔，模型转移仍然是 **1 raw step**，action 为二维相对位移。
可以在 collection 配置中增加 num_branches。

从原始 zarr 读取 episode 初始状态和动作，重新模拟主轨迹与分支。
每个分支重置相同 seed，并重放已经执行的完整动作前缀；检查位置、速度、
角速度、力等动力学状态及当前图像一致，避免只恢复位置却丢失动量。
完全相同的未来图像作为多正样本，不强迫模型区分不可观测的相同后果。
按 source episode 分 train/val/test，不把同一起点的分支拆到不同 split。
新数据与旧数据隔离；不需要删除原数据，但不能使用旧版 NPZ 代替。
采集中的前缀重放比旧版格式转换慢；相同配置可再次运行以跳过已完成 episode。

数据流（D=128，默认 K=2）：

| 环节 | shape |
|---|---|
| 当前观测 / K 个真实未来 | B×3×96×96 / B×K×3×96×96 |
| 共享 CNN 编码 | 当前 O: B×128；真实未来 Y: B×K×128 |
| 动作及共享 action MLP | B×K×2 → B×K×32 |
| 条件转移均值 μ=O+MLP(O,A) | B×K×128 |
| 条件标准差 σ | B×K×128，范围 [0.05,5] |
| 条件匹配矩阵 | B×K×K |

Encoder 输出确定性向量，不输出概率分布。
Transition 定义 p(Y|O,A)=N(μ(O,A),diag(σ²(O,A)))。

## 损失

设 ell_ij = log p(Y_j|O,A_i)/D，S_ij=ell_ij/τ，默认 τ=0.1。
每一行仅在同一起点的 K 个真实后果之间匹配：

```
L_nll = mean_i(-ell_ii)
L_matching = mean_i(logsumexp_j S_ij - logsumexp_{j in positives(i)} S_ij)
L_total = L_nll + L_matching + 0.09 * L_SIGReg
```

没有列方向 CE，没有重建、KL 或额外 MSE 监督。
prediction_mse 是诊断指标，不参与优化。
按维度平均 log-density 是明确的尺度选择；匹配相当于对条件密度做温度缩放。
连续概率密度可以大于 1，所以 NLL 可以为负，不是实现错误。
SIGReg 复用工程实现，仅作用于当前帧和所有真实未来的 encoder latent，
不作用于 action embedding、预测均值或 σ。全体 latent 常数仍然会被 SIGReg 惩罚；
这并不保证实验一定训练成功。
所有权重在 loss YAML 中可设置为 0 做消融。

## 验证、planning 与保存

默认训练 100 epoch，每 5 epoch validation，最后一个 epoch 也验证。
记录 NLL、matching loss/accuracy/gap、SIGReg、positive cosine、prediction MSE、
latent std/effective rank/pairwise cosine、σ及边界饱和率、O/A shuffle、
多 horizon retrieval（预测、identity、打乱动作对照）和 PCA。
匹配准确率将并列分数判为失败；全体后果相同的无信息行不计入准确率，另记录比例。
因此要同时查看 informative_rows，不能仅看 accuracy。

保存 latest.pt（每 epoch，含优化器等）、best_matching.pt、best_loss.pt。
结束后仅运行一次 planning，用 **validation matching loss 选择的权重**，
在 test episodes 上评测；test 结果不参与选择最佳模型。
默认 50 episodes，保存前 5 个视频。planning_evaluated.pt 记录所用权重和结果。

当前 CEM/MPC：horizon=25 raw steps，300 candidates，30 iterations，30 elites，
执行 25 raw actions 后重规划，评测预算 50，action scale=100。
在 train-only 标准化 action 坐标搜索，不额外 clipping；送入原 Predictor/环境前反标准化。
rollout 使用条件均值 μ，不采样 σ；目标成本改为 latent squared error sum。
训练仍是单 raw-step NLL + matching + SIGReg，不改 Encoder；完整边界与旧 YAML
更新方式见 [LEWORLD_PLANNING_ALIGNMENT.md](LEWORLD_PLANNING_ALIGNMENT.md)。
goal 是测试轨迹中起点后 +25 raw step 的固定观测，成功判定复用当前 evaluator，
并不改为“最终标准 T 区域”的另一种任务。
这版环境观测不绘制目标标记，确保训练/当前/goal 的输入渲染一致；
视频左侧是真实执行，右侧是本 episode 固定目标画面。这两个可选渲染开关
不改变旧模型默认行为，也不改变 CEM 和成功判定。

W&B 记录标量和诊断图；**不上传 planning 视频**。
Drive 路径：

```
/content/drive/MyDrive/ACWM/action_matching_v2/data/
/content/drive/MyDrive/ACWM/action_matching_v2/outputs/checkpoints/
/content/drive/MyDrive/ACWM/action_matching_v2/outputs/videos/final_test/
```

## Colab 命令

先启用 GPU 并挂载 Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

已有 clone 时：

```python
%cd /content/ACWM
!git pull --ff-only
!pip install -e '.[colab]'
!wandb login
!python scripts/prepare_action_matching.py \
  --output-root /content/drive/MyDrive/ACWM \
  --config-output configs/colab_action_matching_v2.yaml
!python train_action_matching.py --config configs/colab_action_matching_v2.yaml
```

首次 clone 使用 `!git clone https://github.com/StrawberryChen/ACWM.git /content/ACWM`。
数据准备会生成 GPU、数据路径和 Drive 输出路径配置，无需手动填写路径。
训练命令默认已包含 5 epoch validation 和训练结束后的 planning。

恢复训练：

```python
!python train_action_matching.py --config configs/colab_action_matching_v2.yaml \
  --checkpoint /content/drive/MyDrive/ACWM/action_matching_v2/outputs/checkpoints/latest.pt --resume
```

独立 validation / test evaluation：

```python
!python train_action_matching.py --config configs/colab_action_matching_v2.yaml \
  --mode evaluate --split val \
  --checkpoint /content/drive/MyDrive/ACWM/action_matching_v2/outputs/checkpoints/best_matching.pt
```

如需重新运行 planning（正常训练已自动执行）：

```python
!python train_action_matching.py --config configs/colab_action_matching_v2.yaml \
  --mode plan --split test \
  --checkpoint /content/drive/MyDrive/ACWM/action_matching_v2/outputs/checkpoints/best_matching.pt
```

`--validation-interval 5` 可显式覆盖间隔；`--skip-final-planning` 只用于调试。
