# HW2：微调 ImageNet 预训练模型实现花卉分类

基于 Oxford 102 Flowers Dataset 的花卉分类实验。

本项目实现了：

- ResNet34 微调（Fine-tuning）
- ResNet34 从零训练（Train from Scratch）
- Swin-T / ViT-Tiny Transformer 微调
- YAML 配置超参数搜索
- WandB 日志记录
- 自动保存最佳 checkpoint
- Loss / Accuracy 曲线可视化

---

# 0. 环境准备


安装依赖：

```bash
pip install torch torchvision timm wandb pyyaml tqdm matplotlib scipy
```

---

# 1. 下载数据集

下载：

- 图片
- labels
- 数据划分

数据集地址：

https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

---

# 2. 下载训练好的 checkpoint

下载地址：

https://drive.google.com/drive/folders/1-mhUBnkBCqsu2Z9Z0GP2lMA0ZpR5DvkA?usp=drive_link

---

# 3. 克隆项目

```bash
git clone https://github.com/groolegend/hw2_task1.git
cd   hw2_task1/flower_classification
```

---

# 4. 项目目录结构

确保目录结构如下：

```text
flower_classification
│
├── ckpts
│   ├── resnet_ckpt
│   │   ├── finetune
│   │   │   └── xxx.pth
│   │   │
│   │   └── train_from_scratch
│   │       └── xxx.pth
│   │
│   └── vit_ckpt
│       └── xxx.pth
│
├── data
│   ├── jpg
│   │   └── image_00001.jpg
│   │
│   ├── imagelabels.mat
│   ├── setid.mat
│   └── cat_to_name.json
│
├── train_resnet.py
├── train_transformer.py
├── train_resnet.yml
├── train_transformer.yml
├── eval_resnet.py
├── eval_transformer.py
└── ...
```

---

# 5. 训练 ResNet

修改：

```text
train_resnet.yml
```

## 参数说明

| 参数 | 作用 |
|---|---|
| `pretrained` | 是否使用 ImageNet 预训练权重 |
| `optimizer` | 优化器类型，支持 `adamw` / `sgd` |
| `epochs` | 训练轮数 |
| `backbone_lr` | 主干网络学习率 |
| `fc_lr` | 分类头学习率 |
| `weight_decay` | 权重衰减 |
| `label_smoothing` | 标签平滑 |
| `scheduler` | 调度器，支持 `cosine` / `step` / `none` |
| `momentum` | SGD 动量参数 |
| `eta_min` | Cosine scheduler 最小学习率 |
| `step_size` | StepLR 衰减步长 |
| `gamma` | StepLR 学习率衰减系数 |

---

## 开始训练

```bash
python train_resnet.py
```

---

# 6. 训练 Transformer

修改：

```text
train_transformer.yml
```

## Transformer 参数说明

| 参数 | 作用 |
|---|---|
| `model_name` |支持 `swin_t`|
| `optimizer` | 优化器，支持adamw |
| `epochs` | 训练轮数 |
| `backbone_lr` | Transformer backbone 学习率 |
| `head_lr` | Transformer 分类头学习率 |
| `weight_decay` | 权重衰减 |
| `label_smoothing` | 标签平滑 |
| `scheduler` | 学习率调度器 |
| `eta_min` | Cosine scheduler 最小学习率 |
| `step_size` | StepLR 衰减步长 |
| `gamma` | StepLR 学习率衰减系数 |

---
## 开始训练

```bash
python train_transformer.py
```

---

# 7. 测试模型

## 测试 ResNet

```bash
python eval_resnet.py
```

---

## 测试 Transformer

```bash
python eval_transformer.py
```

---

# 8. 训练输出

训练过程中会自动：

- 保存最佳 checkpoint
- 保存 history.json
- 保存 WandB 日志
- 生成 Loss / Accuracy 曲线

---

