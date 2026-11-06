# Continuous trajectory：独立 planning 诊断

入口：`python -u -m experiments.continuous_dynamics.planning`。
只支持 `continuous_trajectory` 的 action_only / state_action 权重，不重新训练，
不修改 encoder、方向网络、loss、H 或 K。

## 任务与目标

默认从 validation NPZ 用 seed42 固定抽10个案例：当前 dataset state 为起点，
同一 episode +16个 **raw transitions** 的 state 为目标。不是固定绿色 T 的最终任务。
不动/随机/CEM 三个 policy 使用同一批起点和目标，记录 `cases.json`。
不同 checkpoint 对比必须使用相同 split/data/seed/goal_offset。
这是小规模诊断，不宣称复现官方 LeWorld 完整 benchmark。

复用工程现有 `swm/PushT-v1` reset、relative 控制与 success 定义：
success 由真实环境 terminated 或 info.is_success 判断，不是模型 cost 小就成功。
state labels 只用于环境 reset/evaluation，模型不读取未来图像或 simulator state。
目标观察通过同一个已训练 Encoder 得到 goal latent，**只进入 cost，不进入 dynamics/context/direction**。
绿色 marker 的绘制保持训练中的固定 pose，不改变物理 goal_state。

## CEM + MPC

1. Encoder 编码真实当前画面，每次 MPC 更新。
2. 采样候选 `[N,H,2]` 相对动作，H 始终使用 checkpoint 的训练 horizon。
3. 每个候选、每轮 CEM 都重新调用 model.rollout，独立生成 context 和方向箭头。
4. 每个 raw action interval 内进行 checkpoint 的 K 次 Euler integration。
5. Cost=`mean_192((predicted_latent_at_H - goal_latent)^2)`；不加 progress/物理真值/额外平滑 loss。
6. 选 elite 更新均值/标准差，保留 **实际评估过的最佳序列**，不返回未经评估的最终均值。
7. 执行前1个动作，重新观测编码、重新生成候选箭头、重新规划。剩余动作移位作 warm start。

推理无 Gaussian flow noise；CEM 的随机数只用于候选动作搜索。
`model.eval()` / no_grad；方向 head、Encoder 均不会更新。

## 动作单位

数据动作 A_model=(target-agent)/dataset_action_scale，通常 /512。
现有 swm relative 适配约定 target=agent+100*A_env。

因此 `A_env=A_model*dataset_action_scale/100`。
读取实际 env.action_space bounds 并换算为模型单位，与训练输入支持 [-1,1] 取交集后搜索。
拒绝实质性越界执行，不让“模型预测的动作”和“实际截断后的动作”不一致。
`metadata.json`、episode JSON 记录单位和有效 bounds，便于核对。

## 配置与规模

`configs/continuous_trajectory_planning.yaml`：

- 默认 episodes10、max_steps100、goal_offset_steps16；独立于训练 validation.interval。
- N128、elite16、CEM iterations3、执行1步、初始 std .03/min_std .003（模型单位）。
- policies `[cem, zero, random]`，三组均保存前10个 episode 视频。
- 若只看 CEM，可以在该 YAML 改 `policies: [cem]`，但首次建议保留对照。
- 规划比普通 prediction validation 慢；每次 CEM 输出代价和耗时，并显示真实环境执行进度。

## Colab：不用 ROOT/RUN 临时变量，直接选择已有实验

挂载 Drive，准备好仓库后更新并安装依赖：

```python
%cd /content/ACWM
!git pull --ff-only origin main
%pip install -q -e ".[colab]"

```

Colab 的选择放在 notebook 本身（不要在 `!python` 子进程中 input，部分运行时会 EOF）：

```python
from experiments.continuous_dynamics.colab import evaluate_run
evaluate_run("planning", episodes=1, max_steps=5)
```

上述单元格每次会自己找目录，不依赖旧的 RUN/ROOT 变量。
评估读取权重内的训练配置，不依赖同目录的 config.yaml。
确认一例 smoke test 正常后，运行 `evaluate_run("planning")`，默认10案例。
普通终端也可直接用 `python -u -m experiments.continuous_dynamics.planning --choose`。
运行参数从该实验的 saved config.yaml 读取，不用重新生成数据配置，不用重做数据，不用登录 W&B。
默认 val split 用于开发诊断；定稿后可 `--split test`，不要用 test 反复调参。

自定义路径：

```bash
python -u -m experiments.continuous_dynamics.planning \
  --run /完整的实验目录 \
  --planning-config configs/continuous_trajectory_planning.yaml
```

`--run` 默认选择 checkpoints/latest.pt。可以通过 --checkpoint 显式指定同一模型的 best_loss.pt。
自定义目录不在自动发现模式中时，直接用 --run；从未生成过临时 YAML 的新 Colab 也能用。

## 输出

默认保存到所选实验的 `planning/run_时间戳/`（每次新目录），不上传视频到 W&B：

```
metadata.json       # checkpoint SHA256、模型/动作单位、规划参数、split、success/cost说明
cases.json          # 起点与目标，三种 policy 共用
metrics.json        # 每个 policy 的 success_rate_percent、真实 final/best distance 和完整记录
cem/
  planning_episode_000.mp4   # LEFT真实执行，RIGHT固定目标观察
  episode_000.json          # 实际动作、真实距离、每次规划代价/耗时
  episode_000_plan_0001.npz # 搜索序列、整条预测latent、真实执行后的latent、goal latent
zero/...
random/...
```

同时观察：

- 真实 success、final/best goal distance 是否优于 zero/random。
- CEM terminal cost 与真实环境距离是否一致；cost低但真实效果差是 model exploitation 的警讯。
- executed_endpoint_mse：每次真正执行的动作对应的预测 endpoint，与重新观察编码后的误差。
- mean_executed_displacement_pixels：执行的相对 target displacement，而不是 agent 实际移动速度。

final/best distance 是工程现有 swm reward 的负值；包含环境定义的状态尺度，
不要擅自解释成纯 block 欧氏像素距离。视频目标也不是固定绿色 marker。

## 测试范围

单元测试覆盖两个真实 trajectory predictor 的逐候选箭头重算、CEM bounds/确定性/实际最佳cost、
模型到环境单位转换、同一案例的三种 policy、mock环境闭环/JSON/视频接口，以及 --choose 自动入口。
mock环境测试 **不证明真实 Push-T 物理兼容或成功率**；先在 Colab 跑单例 smoke test。
