# 逐时间步方向箭头：原全局箭头实验的独立修正版

旧配置 `continuous_trajectory_action.yaml` / `continuous_trajectory_state_action.yaml`
继续保持原来的全局箭头行为，旧 checkpoint 不被覆盖。

新配置：

- `configs/continuous_trajectory_action_per_step.yaml`
- `configs/continuous_trajectory_state_action_per_step.yaml`

显式 `model.direction_scope: per_step`，输出 **[B,H,192]**，而不是 [B,192]。
同一个 batch 中，每个样本、每个真实 action interval 都有自己的方向。

## 编码与方向

完整候选动作序列 `[B,H,2]` 经共享 Action MLP、位置编码和 bidirectional Transformer，
得到 contextual tokens `[B,H,192]`。**Direction Head 前不做时间 pooling**。
每个 token c_h 已包含该动作在整个序列里的上下文，不是独立编码后忽略动作顺序。

Action-only：

\[
q_h=Head(c_h),\quad d_h=q_h/(\|q_h\|+\epsilon)
\]

State + Action：

\[
q_h=Head([\hat O_h,c_h]),\quad d_h=q_h/(\|q_h\|+\epsilon)
\]

这里 \hat O_h 是 rollout 到该时间步的 **预测当前状态**，不是 E(真实未来帧)。
整个序列初始状态仍为 O_0=E(X_t)。不会通过方向分支偷看真实未来观测。

速度场维持原来的 MLP：concat(当前积分 latent、当前 raw action embedding、
整段动作 context 的 mean、当前 d_h) → V。为隔离这次修改，context mean 仍用于 dynamics 条件，
但它不再被用作每一步的方向 token。单一 global mean 不代表所有箭头相同。

每个 h 内的 K=4 次积分共用 d_h；进入下一个真实时间步后换成 d_{h+1}。
没有任何 loss 要求 d_h 和 d_{h+1} 同向，允许跨真实区间转弯、反向。

\[
L_{prog}=mean_{B,H,K}\max(0,m-V^{h,k}\cdot stopgrad(d_h))
\]

Prediction loss 仍然是完整 open-loop trajectory 与真实未来 latent 的 MSE。
模型没有加入新 loss、teacher、SIGReg、BN 或物理监督。
未 detach 的 d_h 同时进入速度场，方向网络从两项 loss 的 dynamics 路径收到梯度。

两种 scope 使用完全相同的参数形状和初始化；区别是上下文 token 与方向计算的时间粒度。
依旧只是研究假设，局部方向约束不构成数学上的防坍缩保证。

## 启动

复用 raw-step 数据，不必重新准备；需要从头训练，不能 --resume 全局箭头权重。
两个 YAML 已固定使用你这次的数据：

`/content/drive/MyDrive/ACWM/continuous_trajectory_fresh_20261104_190724/avs_target/data/{train,val,test}/episode_*.npz`

无需 ROOT、RUN、DATA_CONFIG，也无需生成临时 YAML 或传 --data-config。
若以后迁移数据，只修改对应 YAML 的三个 data.*_paths。两个版本均独立列出模型、训练、
loss、validation、probe 和日志参数，改 Action-only 不会隐式改变 State+Action。

```bash
python -m experiments.continuous_dynamics.train \
  --config configs/continuous_trajectory_action_per_step.yaml

python -m experiments.continuous_dynamics.train \
  --config configs/continuous_trajectory_state_action_per_step.yaml
```

H16、K4、100 epochs、每5轮验证仍然保持。loss.weight、margin 可分别在新 YAML 改。
做 global/per_step 公平对照时，复制同一组 H/K/weight/margin/seed/data 设置，
不要在改方向粒度的同时调整其他参数。

验证/probe/planning 入口不变，使用新实验各自保存的 config.yaml 和 checkpoint。
Probe 新增数据加载提示、特征提取进度条和 MLP 训练进度，不再全程静默。
规划每个候选序列重新生成完整 d_0...d_{H-1}，每次 MPC 重新观察后重新计算。
旧全局模型与新逐步模型 checkpoint 有配置隔离，避免同形状权重被错误当成同一方法续训。

## 诊断与测试

- `direction/temporal_cosine`：相邻真实时间步箭头 cosine，仅诊断，不加入 loss。
- progress/violation 与 velocity cosine 均针对对应的 d_h，不会错误广播为同一根箭头。
- `visualizations/latents_*.npz` 保存 `[B,H,192]` 方向供后续分析。
- 测试两种模式逐步 head 梯度、真实未来无泄漏、预测状态反馈、turn/reversal loss、
  full rollout、validation/probe/plot/CEM 适配和新旧权重隔离。

```bash
python -m unittest tests.test_continuous_per_step -v
```

## 新 Colab 运行时

运行时内存和 /content 是临时的，所以仍需挂载 Drive、恢复仓库、安装依赖和（训练在线记录时）登录 W&B。
这些是会话初始化，不是重新准备数据或生成实验配置：

```python
from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import subprocess
if not Path("/content/ACWM/.git").exists():
    subprocess.run(["git", "clone", "https://github.com/StrawberryChen/ACWM.git", "/content/ACWM"], check=True)
%cd /content/ACWM
!git pull --ff-only origin main
%pip install -q -e ".[colab]"
import wandb
wandb.login()
```

之后直接执行上面的训练命令。已有 latest.pt 的同一目录不能无意重新训练；
新参数实验先改 YAML 的 output_dir 和 wandb.name。续训使用该实验保存的 config.yaml，
不能在改变模型或loss后 --resume 旧权重。
规划 swm 环境要求 Pymunk 7（与旧 gym-pusht 的6系API不同）；若使用新运行时做 planning，
安装额外依赖后执行 `!python -m pip install --upgrade "pymunk>=7,<8"`。
