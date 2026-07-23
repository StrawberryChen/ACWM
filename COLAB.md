# Colab 训练清单

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone <YOUR_REPOSITORY_URL> ACWM
%cd /content/ACWM
!pip install -e '.[colab]'
!wandb login
```

The `colab` extra installs LeWorld's `stable-worldmodel` package so planning
validation can use `swm/PushT-v1`, the same Push-T environment wrapper used by
LeWorld. The older `gym_pusht/PushT-v0` path is kept only for legacy configs.

将 `configs/default.yaml` 中的 `data.train_paths`、`data.val_paths` 改为 Google
Drive 上的绝对路径。训练集与验证集应按 trajectory 分开，不要将同一轨迹的窗口
随机拆到两边。

更快捷的做法是直接下载并转换原始 Diffusion Policy Push-T zarr 数据。下面的命令会按
episode 划分训练/验证集、写入 Google Drive，并自动生成 `configs/colab_acwm_v3_n1.yaml`。
转换后的 NPZ 会保存 `frames`、LeWorld 对齐后的相对动作 `actions`，以及 planning 所需的
5D simulator states：

```text
states = [agent_x, agent_y, block_x, block_y, block_angle]
actions = clip((absolute_target_xy - agent_xy) / 100, -1, 1)
```

这里的 action 语义对齐 `swm/PushT-v1`：环境内部执行
`target_xy = agent_xy + action * 100`。

```bash
!python scripts/prepare_pusht_zarr.py --output-root /content/drive/MyDrive/ACWM
```

生成的 `configs/colab.yaml` 会自动使用 `device: cuda`。请先在 Colab 选择
`Runtime → Change runtime type → T4 GPU`；程序也会在启动时检查 GPU 是否可用。

首次创建模型时会按 LeWorld 风格从头初始化 ViT-Tiny patch14 和 BatchNorm
projector；不会下载预训练 ViT 权重。旧 checkpoint 与新结构不兼容，修改后需要从头训练。
如果你之前准备过旧版数据，也必须重新运行 `scripts/prepare_pusht_zarr.py`。旧数据里的 action
大概率是 `[0,512]` 绝对目标点；新环境 `swm/PushT-v1` 需要 `[-1,1]` 相对动作，不重新准备会导致
训练和 planning 的 action 语义错位。

```bash
!python train.py --config configs/colab_acwm_v3_n1.yaml
```

运行期间可以在 W&B 查看：

- 每 epoch 的 train / validation loss；
- prediction validation GIF；
- 每 10 epoch 的 50-episode planning success rate；
- Push-T 闭环规划 MP4。

`outputs/acwm.pt` 是每个 epoch 更新的 latest checkpoint；
`outputs/acwm_best.pt` 只在 planning success rate 创新高时更新。建议将 `outputs/`
指向或复制到 Google Drive，避免 Colab runtime 回收后丢失。
