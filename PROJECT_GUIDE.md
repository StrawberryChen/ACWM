# ACWM 项目入口

当前主实验是 Continuous Trajectory。Colab 使用 [顺序 Notebook](notebooks/ACWM_Trajectory_Colab.ipynb)，
或从仓库根目录用 [工作流配置](configs/colab_trajectory.yaml) 选择 global/per-step 和 action-only/state+action。
Notebook 参数持久化在各次实验的 `workflow.yaml`，底层有效参数仍随 `latest.pt` 保存。
切换模型、数据或训练参数时换实验名；继续原实验则保留原参数。旧代码和旧权重不迁移、不覆盖。

| 实验 | 入口与说明 |
| --- | --- |
| Continuous trajectory（当前主线） | `experiments.continuous_dynamics.workflow`、[per-step](CONTINUOUS_PER_STEP.md)、[planning](CONTINUOUS_PLANNING.md) |
| 单步 continuous dynamics | [诊断与训练](CONTINUOUS_DYNAMICS.md) |
| ACWM / LeWorld / Forward-Inverse / Motion Token | `train.py`，配置在 `configs/`；见 [项目说明](README.md) |
| Action-Contrastive / Action Matching | `train_action_contrastive.py` / `train_action_matching.py`；[第一版](ACTION_CONTRASTIVE.md)、[第二版](ACTION_MATCHING_V2.md) |
| Observation Graph / Edge-Guided | `train_obs_graph.py` / `train_edge_guided.py`；各自的配置在 `configs/` |
| AVS-Target / AVS-Terminal | `experiments.avs_target`；[AVS Target](AVS_TARGET.md)、[AVS Terminal](AVS_TERMINAL.md) |

## 当前 Notebook 的运行顺序

1. 初始化：挂载 Drive、更新仓库、装 trajectory 专用依赖、读取可选的 Colab Secret。
2. 参数：选择实验名和参数，检查 GPU、训练/验证/测试数据。
3. 第一轮：训练 1 轮并暂停，保存完整可续训权重；100 轮目标不会变成 1 轮。
4. Drive 验收：从网页确认 `checkpoints/latest.pt` 可见，勾选后续训；中断后可接最近保存的轮次。
5. Probe、6. Planning：读取本次最后一轮权重，无需搜索/手填路径；结果在各自的 Drive 子目录。

权重写临时文件、flush/fsync、同目录替换；第 1 轮和本次停止轮会从挂载文件系统读回核验。
挂载层读回不等于 Drive 云端验收，所以第 4 步仍要求确认。W&B 只记指标和图，不上传权重。
最新权重每轮更新，`best_loss.pt` 只在验证改善时更新。
新流程复用已有 raw-step NPZ；缺数据时参考 [现有数据准备命令](CONTINUOUS_TRAJECTORY.md)，不要清空已有数据。
专用的 `.[colab-trajectory]` 用 Pymunk 7；旧 gym-pusht 实验的 Pymunk 6 依赖应在另一会话安装。
