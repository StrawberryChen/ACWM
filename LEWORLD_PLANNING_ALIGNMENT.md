# 当前 raw-step 实验的 LeWorld planning 对齐

适用：action_contrastive / action_matching_v2。训练、Encoder、loss 和已有权重不变。

官方参考：
- https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml
- https://github.com/lucas-maes/le-wm/blob/main/config/eval/solver/cem.yaml
- https://github.com/lucas-maes/le-wm/blob/main/jepa.py
- https://github.com/galilai-group/stable-worldmodel/blob/main/stable_worldmodel/planning/solver/cem.py

| 项目 | 官方 Push-T | 当前单步模型适配 |
|---|---|---|
| 搜索变量 | 5 blocks × 10 dimensions | 25 raw steps × 2 dimensions |
| raw action 总数 | 25 | 25 |
| 执行后重规划 | 5 blocks = 25 raw steps | 25 raw steps |
| 评测预算 | 50 raw steps | 50 raw steps |
| 固定 dataset goal | 起点后 25 raw steps | 起点后 25 raw steps |
| candidates / elites / iterations | 300 / 30 / 30 | 300 / 30 / 30 |
| 终点成本 | latent squared error sum | latent squared error sum |
| action 搜索坐标 | dataset StandardScaler | 本模型 train trajectories 的 population mean/std |
| 搜索动作裁剪 | 无 | 无 |
| 采样 seed | 42 | 42，独立 RNG |

五动作块展平后是相同数量的独立搜索变量，但模型计算不同：官方一个动作块
直接预测五步之后的 latent，本实验仍调用单 raw-step Predictor 25 次。
这里 action_block 必须保持 1，不能仅填 5 导致接口误以为已训练了 10D Action Encoder。
统计量只拟合 train，不读取 test 以确定动作分布；常数维度 scale=1。
候选反标准化后送给原 Predictor，最终返回真实 relative actions [B,25,2] 给环境。
不重复同一个动作 5 次，不把高斯均值 Head 改成其他结构。

normalized action 可能超过 +/-1，亦可能产生超出旧训练数据范围的相对动作。
这是移除额外 CEM clipping 的结果；该设置不保证模型外推准确。
cem_action_* 指标为标准化搜索坐标，raw_action_* 为环境 relative action 坐标。
新增 W&B 记录 horizon、重规划长度、预算、成本类型、clipping 开关及 action mean/std。

本修改不声称整个实验与 LeWorld 完全一致：保留 96px CNN 输入、原渲染选项、
原数据 split/来源、单帧状态模型，以及已有训练数据中动作 clipping。
官方 Encoder 是 224px ViT，训练 action 数据与我们的 NPZ 不相同，因此标准化
数值也不保证相同。五动作打包的训练需要另改 dataset/Predictor，不能仅改 planning 配置。

## 已有 Colab YAML

git pull 不会自动更新以前生成的 colab YAML。保留旧数据和 Drive 输出路径，运行：

```python
!python scripts/align_planning_config.py --config configs/colab_action_contrastive.yaml
# 或第二版：
!python scripts/align_planning_config.py --config configs/colab_action_matching_v2.yaml
```

然后使用原 train/evaluate/plan 命令。可以直接用现有 checkpoint 重新 planning，
不必重新训练或准备数据。新采集的第二版数据仍按其采集 manifest 管理，
为了改 planning 不应重新运行采集脚本。
默认 validation 间隔、planning 开关/最终运行方式、视频 Drive 路径保持不变。
