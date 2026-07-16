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

The `colab` extra pins Pymunk below version 7 because `gym-pusht 0.1.x` still uses
the collision-handler API removed by Pymunk 7.

将 `configs/default.yaml` 中的 `data.train_paths`、`data.val_paths` 改为 Google
Drive 上的绝对路径。训练集与验证集应按 trajectory 分开，不要将同一轨迹的窗口
随机拆到两边。

更快捷的做法是直接下载并转换原始 Diffusion Policy Push-T zarr 数据。下面的命令会按
episode 划分训练/验证集、写入 Google Drive，并自动生成 `configs/colab.yaml`。转换后的
NPZ 会保存 `frames`、`actions` 和 LeWorld-style planning 所需的完整 5D simulator
`states = [agent_x, agent_y, block_x, block_y, block_angle]`：

```bash
!python scripts/prepare_pusht_zarr.py --output-root /content/drive/MyDrive/ACWM
```

生成的 `configs/colab.yaml` 会自动使用 `device: cuda`。请先在 Colab 选择
`Runtime → Change runtime type → T4 GPU`；程序也会在启动时检查 GPU 是否可用。

首次创建模型时会按 LeWorld 风格从头初始化 ViT-Tiny patch14 和 BatchNorm
projector；不会下载预训练 ViT 权重。旧 checkpoint 与新结构不兼容，修改后需要从头训练。
如果你之前准备过旧版数据，也需要重新运行 `scripts/prepare_pusht_zarr.py`，否则 planning eval
会因为 NPZ 缺少 `states` 而停止。

```bash
!python train.py --config configs/colab.yaml
```

运行期间可以在 W&B 查看：

- 每 epoch 的 train / validation loss；
- prediction validation GIF；
- 每 50 epoch 的 50-episode planning success rate；
- Push-T 闭环规划 MP4。

`outputs/acwm.pt` 是每个 epoch 更新的 latest checkpoint；
`outputs/acwm_best.pt` 只在 planning success rate 创新高时更新。建议将 `outputs/`
指向或复制到 Google Drive，避免 Colab runtime 回收后丢失。
