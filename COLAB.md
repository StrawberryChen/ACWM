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

更快捷的做法是直接下载并转换官方 LeRobot Push-T 数据。下面的命令会按 episode
划分训练/验证集、写入 Google Drive，并自动生成 `configs/colab.yaml`：

```bash
!python scripts/prepare_pusht.py --output-root /content/drive/MyDrive/ACWM
```

```bash
!python train.py --config configs/colab.yaml
```

运行期间可以在 W&B 查看：

- 每 epoch 的 train / validation loss；
- prediction validation GIF；
- 每 5 epoch 的 100-episode planning success rate；
- Push-T 闭环规划 MP4。

`outputs/acwm.pt` 是每个 epoch 更新的 latest checkpoint；
`outputs/acwm_best.pt` 只在 planning success rate 创新高时更新。建议将 `outputs/`
指向或复制到 Google Drive，避免 Colab runtime 回收后丢失。
