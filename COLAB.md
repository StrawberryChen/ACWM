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

将 `configs/default.yaml` 中的 `data.train_paths`、`data.val_paths` 改为 Google
Drive 上的绝对路径。训练集与验证集应按 trajectory 分开，不要将同一轨迹的窗口
随机拆到两边。

```bash
!python train.py --config configs/default.yaml
```

运行期间可以在 W&B 查看：

- 每 epoch 的 train / validation loss；
- prediction validation GIF；
- 每 5 epoch 的 100-episode planning success rate；
- Push-T 闭环规划 MP4。

`outputs/acwm.pt` 是每个 epoch 更新的 latest checkpoint；
`outputs/acwm_best.pt` 只在 planning success rate 创新高时更新。建议将 `outputs/`
指向或复制到 Google Drive，避免 Colab runtime 回收后丢失。
