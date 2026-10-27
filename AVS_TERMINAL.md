# AVS-Target terminal measurement extension

在现有 `experiments/avs_target/` 上增量扩展。原 AVS 算子、Flow Predictor 和训练入口保留。

## 三个实验

| 配置 | 训练目标 | 入口 | Goal planning |
|---|---|---|---|
| configs/avs_only.yaml | 原 AVS-only FM | experiments.avs_target.train | 不支持 |
| configs/terminal_only.yaml | Terminal FM | experiments.avs_target.joint_train | 诊断审核后显式启用 |
| configs/avs_terminal.yaml | Motion FM + Terminal FM | experiments.avs_target.joint_train | 诊断审核后显式启用 |

`avs_only.yaml` 继承原 `avs_flow.yaml`，解析结果完全一致。已有 AVS-only 权重、统计量、训练命令不变。
新联合模型与原 AVS-only 权重结构不同，使用独立目录，不宣称可直接加载旧模型权重。

## 固定 terminal operator

代码：`experiments/avs_target/terminal.py`。

- 默认 downsample：RGB [B,3,96,96] → 固定 area resize → [B,3,24,24] → [B,1728]。
- 可选 dct：先固定 area resize，再做正交 DCT-II，保留 RGB 各 8×8 左上角低/中频系数，D=192。
- DCT 系数固定除以 resize size，以稳定尺度；inverse DCT 只用于显示截断后的近似图。
- 无 Parameter、无 CNN/ViT、无预训练、无可训练归一化，全程 no_grad。
- RGB 使用 [0,1]，不拟合任何 terminal target encoder；operator spec 与固定 buffers 保存进 checkpoint。
- Goal image 调用完全相同的 operator 和固定尺度；绝不调用 CSE 得到 goal latent。
- Random projection terminal target 暂未实现，不作为默认 backend。

## 数据流 / shapes（默认 H=16）

```
X_t [B,3,96,96] -> 原随机 CNN CSE -> O [B,128]
A [B,16,2] -> 共享 Action Encoder -> [B,16,128] -> [B,16,256]

真实 frames [B,17,3,96,96]
  -> 原 Fixed AVS -> M [B,16,288]（联合版本）
  -> 取 frames[:,-1] -> Fixed Terminal -> G [B,1728]

独立 Gaussian noise epsilon_M / epsilon_G；同一 flow time s [B]
  -> noisy M [B,16,288] / noisy G [B,1728]
  -> 分别投影 256D
  -> 一个共享双向 Transformer
  -> motion velocity head [B,16,288]
  -> terminal velocity head [B,1728]
```

联合模型的 token 序列：[当前 O token, H 个 motion+action tokens, terminal token]，shape [B,H+2,256]。
Terminal-only 不计算 AVS，也没有 motion projection/head；H 个位置只提供 action/time 条件。
两者只有一个 CSE、一个 Transformer，不是两套独立世界模型。
World trajectory position 与 flow time 使用分开的编码；不使用 causal mask。

## Loss

```
X_M(s) = (1-s)*epsilon_M+s*M
X_G(s) = (1-s)*epsilon_G+s*G
V_M_target = M-epsilon_M
V_G_target = G-epsilon_G

L_motion = mean((V_M_pred-V_M_target)**2)
L_terminal = mean((V_G_pred-V_G_target)**2)
L_total = lambda_motion*L_motion + lambda_terminal*L_terminal
```

两个 tensor 各自取 mean，再加权，默认两个 lambda=1。不将高维 tensor 拼接后统一 mean。
Terminal-only 的 motion loss 记录为 0，没有 AVS 输入；应以独立配置比较，而非假装计算了 motion。
无 SIGReg/VICReg、contrastive、latent MSE、goal latent loss 或额外像素重建 loss。
Terminal target 本身来自固定低分辨率图像，但训练监督是 measurement 的 FM velocity，
没有另加一个 trainable decoder 的 reconstruction objective。
梯度只进入 CSE / Action Encoder / 共享 Transformer / velocity heads。

## Prediction-first 验证

默认 100 epoch，每 5 epoch validation，最后一轮也验证。分别记录：
train/val motion_flow_loss、terminal_flow_loss，以及加权 total loss。
保留 collapse 指标、相同 noise/time 下 O/A shuffle 和 s=0/0.1/0.5/0.9 分段依赖性。
从 noise 完整生成时再报告 terminal MSE/cosine、motion MSE/cosine/direction/magnitude（joint）。
K=4/8/16/32，H=1/2/4/8/16；短 H 的 terminal label 正确取 X_t+H，而不是仍取长窗口末帧。
同一 H、不同 K 使用相同 source noise。记录完整生成时的 state/action shuffle terminal MSE delta。

生成图：当前真实帧、真实 terminal frame、固定 terminal measurement、生成 terminal measurement。
联合模型继续保存 motion trajectory 对比视频。PNG 上 W&B，视频只存 Drive。
latest.pt 每 epoch 保存；best_terminal.pt 仅根据 validation 完整 H、最大 Euler K 的 terminal MSE 选择，
这样 Terminal-only 与 AVS+Terminal 用相同选择标准。支持 optimizer/scheduler/RNG 恢复。
terminal spec、AVS 统计量、原始 random CSE 和训练数据 provenance 都保存。

冻结 CSE 的 probe 仍使用原 probe.py；真实 state 只在这个独立 evaluation 阶段使用。
对比 trained/random CSE 和 constant baseline，Linear/MLP 头只用 train 拟合、val 选择。
probe 和独立 evaluate 报告保存 checkpoint SHA256，用于核对规划的权重。

### Colab 数据来源兼容

独立 evaluate 不再要求训练文件的绝对路径/mtime 完全不变，也不会重新拟合 checkpoint 的 target normalization。
修改时间不是数据内容身份；新训练保存流式 SHA256，续训以内容核对，允许相同内容复制/迁移。
旧 checkpoint 没有 hash：续训只能核对原路径和文件大小，忽略 mtime 并明确警告无法证明内容完全相同。
真正不同的数据应使用新 output-dir，不应靠跳过检查继续训练。
独立验证配置另存 evaluation_val/config.yaml，不覆盖训练 config.yaml。
仍严格检查模型结构、固定算子、动作尺度，以及 checkpoint 与 avs_stats.json 的 mean/std 一致。
probe 本身没有训练文件 mtime 校验；应先完成 evaluate，再运行原 probe 命令。

## Colab：先做 Phase 1-5

已有 `configs/colab_avs_flow.yaml` 和 raw-step 数据时，不必重新准备或复制数据：

```python
%cd /content/ACWM
!python -m experiments.avs_target.joint_train \
  --config configs/terminal_only.yaml --data-config configs/colab_avs_flow.yaml
!python -m experiments.avs_target.joint_train \
  --config configs/avs_terminal.yaml --data-config configs/colab_avs_flow.yaml
```

`--data-config` 仅借用数据路径、动作尺度和 device，保留各自模型/loss 配置；
输出放在同一 outputs 根目录的 terminal_only / avs_terminal，不覆盖 avs。
首次准备数据也可使用：

```python
!python scripts/prepare_avs_target.py --base-config configs/avs_terminal.yaml \
  --config-output configs/colab_avs_terminal.yaml
!python -m experiments.avs_target.joint_train --config configs/colab_avs_terminal.yaml
```

训练会保存已解析的完整配置。以下以 Terminal-only 为例，联合实验把目录换为 avs_terminal：

```python
ROOT = "/content/drive/MyDrive/ACWM/avs_target/outputs"
EXP = f"{ROOT}/terminal_only"
!python -m experiments.avs_target.joint_train --config "{EXP}/config.yaml" \
  --mode evaluate --split val --checkpoint "{EXP}/checkpoints/best_terminal.pt"
!python -m experiments.avs_target.probe --config "{EXP}/config.yaml" \
  --checkpoint "{EXP}/checkpoints/best_terminal.pt"
```

probe 只更新分析用 heads，不修改主 checkpoint。恢复训练用 latest.pt --resume。
`evaluation_val/report.json` 和 `probes/metrics.json` 是主要诊断报告。
比较两种模型，不自动宣布研究假设成立：

```python
!python -m experiments.avs_target.compare \
  --terminal-dir "{ROOT}/terminal_only" --joint-dir "{ROOT}/avs_terminal" \
  --output "{ROOT}/terminal_comparison.json"
```

先判断 CSE 尺度/秩、O/A dependency、生成效果、probe 相对 random/constant 是否有价值；
不要用 test 结果选择 checkpoint 或反复调参。

## Phase 6：显式 CEM + MPC

没有在本地完成真实数据训练，也没有证据证明这两种模型已学好。
下面代码可用但默认关闭，训练不会自动运行。必须先人工检查同一 checkpoint 的 val prediction/probe。
命令中的 reviewed 是人工确认，不是程序声称已通过科学有效性检验。
程序只核对报告完整性/数值有效性及 SHA256，不虚构“正常”的统一阈值。

```python
!python -m experiments.avs_target.planning --config "{EXP}/config.yaml" \
  --checkpoint "{EXP}/checkpoints/best_terminal.pt" \
  --diagnostics-report "{EXP}/evaluation_val/report.json" \
  --probe-report "{EXP}/probes/metrics.json" \
  --enable-planning --diagnostics-reviewed --split test
```

默认 planning：512 candidates、64 elites、5 iterations、H=16、R=1、future_samples=1、Euler K=8。
全为 YAML 参数。每个 candidate 生成 motion/terminal，但成本只计算：

`J(A)=mean_k mean_d((G_pred[k,d]-G_goal[d])**2)`。

支持 future_samples>1；一轮 plan 的所有候选和 iterations 复用同一组 K 个 source noises
（common random numbers 降低噪声排序偏差），下一次 MPC plan 重新采样。
按 candidate_batch_size 分块评估，不改变结果。返回搜索中实际评估过的最佳序列，
而非未经评估的 elite mean。随后只执行前 R 个 raw actions，重新观察真实帧，再规划。
不把生成 terminal image 重新编码为当前状态，不复用虚构的 future state latent。

数据动作单位为 (target-agent)/action_scale；swm 环境内部使用 target=agent+100*action。
执行时换算 `env_action=model_action*action_scale/100`；默认512与100不混用，不偷偷额外 clipping。
动作边界在模型归一化单位中配置。超出观测空间/工作台的预测仍可能发生，应观察 raw rollout，
不能把单步训练外推范围当作已验证安全区间。

观察画面默认保留原始 Push-T 数据集的固定 T marker pose [256,256,pi/4]。
绘制 marker 的 pose 与本 episode 的真实物理 goal_state 分开：采样目标不会改变输入背景，
reward/success 仍根据采样 state 计算。自定义数据不含 marker 时，应同步设置 with_target=false；
这不能保证所有数据来源的渲染完全一致，需要检查 terminal target/真实环境对比。

默认50 episodes、100 raw-step预算、dataset 起点后+25 raw-step作为固定目标。
success 使用 swm 的真实状态成功条件，包含 agent 和 block，**不是标准固定 T 区域 coverage**。
final_goal_distance 是 swm reward 对应的真实 state distance；episode_reward 是逐步 reward 之和。
steps_to_success 只对成功 episodes 求均值；无成功时省略，不以0伪装成功。
记录 success_rate（百分比）、final_goal_distance、episode_reward、steps_to_success、cem_time、
world_model_time、candidates_per_second，另记录成功数量和 episode steps。
耗时在 CUDA 上同步测量，world_model_time 是 Flow sampling 时间。

输出：`{EXP}/planning_test/`。真实执行视频左=环境、右=固定 goal；只保存前5个episode视频。
每次 MPC/CEM 最佳 candidate 都保存所有 K 个 terminal samples、动作、goal measurement、cost 的 NPZ，
以及预测 terminal measurement 均值 PNG（DCT显示为固定截断逆变换）。视频不上传 W&B。
Planning test 不选择/覆盖 best checkpoint。

## 新增主要文件

- terminal.py：固定 downsample/DCT。
- joint_model.py：共享双向 Transformer、双 velocity heads、分支 mean FM loss、Euler。
- joint_train.py / joint_evaluate.py：训练/生成评估/续训/terminal visualization。
- planning.py：terminal-only CEM cost、expected cost、MPC、报告审核与真实环境视频。
- compare.py：读取 val prediction/probe 比较报告。
- configs/avs_only.yaml、terminal_only.yaml、avs_terminal.yaml。
- tests/test_avs_terminal.py：算子、梯度、对齐、loss权重、生成、训练恢复、规划与真实环境测试。
