# Lightweight 3D LiDAR Object Detection in Bird's-Eye-View (PyTorch)

An efficient, modular PyTorch implementation of anchor-free 3D object detection from LiDAR point clouds in Bird's-Eye-View (BEV). This repository features multiple backbone architectures including **MobilePIXOR** (lightweight MobileNetV2 inverted residual blocks for edge robotics), classic **PIXOR** (ResNet-style with Feature Pyramid Networks), **RPN**, and **PointPillars** BEV backbones.

---

## 📄 Associated Research Publication

**Lightweight 3D LiDAR Rubber Tree Detection with Knowledge Distillation and Efficient Attention**

*Engineering Research Express* — IOP Publishing

**Paper:** [Read on IOPscience](https://iopscience.iop.org/article/10.1088/2631-8695/aebd41)

**DOI:** [10.1088/2631-8695/aebd41](https://doi.org/10.1088/2631-8695/aebd41)

### Research Overview

This research investigates lightweight 3D LiDAR object detection for rubber tree perception, focusing on efficient attention mechanisms and knowledge distillation.

The repository provides a modular PyTorch framework for Bird's-Eye-View (BEV) object detection, including MobilePIXOR, PIXOR, PointPillars, and RPN backbones.

### Citation

If you use this repository in your research, please cite the associated publication using the official bibliographic information available on IOPscience.

## 🌟 Key Highlights

- **Voxel-based BEV Representation**: Fast occupancy voxelization transforming raw irregular 3D LiDAR point clouds into a structured 2.5D BEV tensor, eliminating expensive 3D convolutions.
- **Anchor-Free Detection Header**: Multi-task convolutional header predicting:
  - Class probability heatmaps (Gaussian focal or binary encoding).
  - Centroid sub-voxel metric offsets $(\Delta x, \Delta y)$.
  - Log bounding box dimensions $(\log w, \log l)$.
  - Continuous double-angle yaw orientation $(\cos 2\theta, \sin 2\theta)$.
- **Lightweight Backbones for Real-Time Inference**:
  - `mobilepixor`: MobileNetV2 depthwise separable and inverted residual blocks with lateral connections.
  - `pixor`: ResNet-style bottleneck blocks with top-down feature aggregation.
  - `pointpillars`: Pillar-based feature extraction backbone.
  - `rpn`: Scalable Region Proposal Network architecture.
- **Rotated Non-Maximum Suppression (NMS)**: Accurate polygon IoU suppression for arbitrarily oriented 3D bounding boxes.
- **Production-Ready Deployment**: Built-in support for **TorchScript (`torch.jit.script`)** export for low-latency deployment in C++ and ROS ecosystems.
- **Interactive 3D Visualizer**: Real-time 3D point cloud and oriented bounding box inspection using OpenGL/VisPy.

---

## 🏗️ Repository Architecture

```
├── configs/                  # Model, dataset, and training configurations
│   ├── mobilepixor.json      # Lightweight MobilePIXOR config
│   ├── pixor.json            # Classic PIXOR config
│   ├── pointpillars.json     # PointPillars config
│   └── rpn.json              # RPN config
├── core/
│   ├── datasets/
│   │   └── dataset.py        # Point cloud voxelizer, data loader & augmentations
│   ├── losses/
│   │   ├── focal_loss.py     # Gaussian modified focal loss and standard focal loss
│   │   ├── l1_loss.py        # Masked L1 and Smooth L1 regression losses
│   │   └── loss_fn.py        # Multi-task detection objective
│   ├── models/
│   │   ├── backbones/        # Modular backbones (mobilepixor, pixor, rpn, pointpillars)
│   │   ├── heads/            # Multi-task CNN prediction heads
│   │   ├── torchscript/      # TorchScript exportable model wrappers
│   │   └── model.py          # Unified detector interface
│   ├── torchplus.py          # PyTorch functional utilities
│   └── train_agent.py        # End-to-end training and validation engine
├── eval/
│   ├── eval.py               # Precision, Recall, and Average Precision (AP) benchmark
│   └── auxiliary.py          # Geometry, scale, and yaw evaluation metrics
├── utils/
│   ├── gaussian.py           # 2D Gaussian heatmap encoding
│   ├── postprocess.py        # Decoding and rotated polygon NMS
│   ├── preprocess.py         # Point cloud voxelization and coordinate transformation
│   ├── split_dataset.py      # Dataset train/val split generator
│   ├── to_torchscript.py     # Script to export PyTorch weights to TorchScript (.pt)
│   └── transform.py          # 3D spatial data augmentations
├── vis/
│   ├── dataset_vis.py        # Interactive 3D scan and ground truth viewer
│   └── test_vis.py           # Interactive 3D detection vs ground truth viewer
├── main.py                   # Main training entrypoint
└── requirements.txt          # Python dependencies
```

---

## 📊 Backbone Comparison

| Backbone | Input Dim (C×H×W) | Stride | Key Components | Target Application |
|---|:---:|:---:|---|---|
| **MobilePIXOR** | $30 \times 480 \times 480$ | 4 | Inverted Residuals, Depthwise Conv, FPN | Edge robotics, embedded GPUs, AGVs |
| **PIXOR** | $30 \times 480 \times 480$ | 4 | ResNet Bottlenecks, FPN | High accuracy BEV detection |
| **PointPillars** | $30 \times 480 \times 480$ | 4 | Top-down ConvBlocks + TransposeConv | Fast multi-scale representation |
| **RPN** | $30 \times 480 \times 480$ | 1 | Multi-stride convolutions | Dense BEV proposal generation |

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/LiDAR-PIXOR-PyTorch.git
cd LiDAR-PIXOR-PyTorch

# Create and activate environment
conda create -n lidar-det python=3.10 -y
conda activate lidar-det

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Organize your LiDAR scans and 3D labels (KITTI format) as follows:

```
data/
└── kitti/
    ├── pointcloud/           # Binary (.bin) LiDAR point clouds [x, y, z, intensity]
    │   ├── 000001.bin
    │   └── ...
    └── label/                # Label text files [Class, h, w, l, x, y, z, yaw]
        ├── 000001.txt
        └── ...
```

Generate train and validation split manifests:

```bash
python utils/split_dataset.py \
    --data_dir data/kitti/pointcloud \
    --save_dir data/list \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --dataset_name kitti
```

### 3. Training

Train **MobilePIXOR** with Gaussian focal loss:

```bash
python main.py --config configs/mobilepixor.json
```

Train **PIXOR**:

```bash
python main.py --config configs/pixor.json
```

Optional CLI overrides:
```bash
python main.py --config configs/mobilepixor.json --epochs 50 --batch_size 8 --device cuda
```

### 4. Evaluation

Evaluate precision, recall, and Average Precision (AP) at various IoU thresholds:

```bash
python eval/eval.py \
    --config configs/mobilepixor.json \
    --checkpoint experiments/mobilepixor_gaussian_baseline/best_checkpoints/best_model.pt \
    --data data/list/val.txt \
    --iou_thresh 0.5
```

### 5. Interactive 3D Visualization

Inspect raw point clouds and ground truth annotations:

```bash
python vis/dataset_vis.py --config configs/mobilepixor.json --data data/list/val.txt
```

Inspect model predictions in 3D:

```bash
python vis/test_vis.py \
    --config configs/mobilepixor.json \
    --checkpoint experiments/mobilepixor_gaussian_baseline/best_checkpoints/best_model.pt \
    --data data/list/val.txt
```

**Viewer Controls:**
- `N` or `Right Arrow`: Next LiDAR scan
- `B` or `Left Arrow`: Previous LiDAR scan
- `G`: Toggle Ground Truth boxes on/off
- `Q` or `Escape`: Quit viewer

### 6. TorchScript Export & Deployment

Export trained PyTorch weights to a standalone TorchScript `.pt` file for C++ and ROS deployment:

```bash
python utils/to_torchscript.py \
    --config configs/mobilepixor.json \
    --checkpoint experiments/mobilepixor_gaussian_baseline/best_checkpoints/best_model.pt \
    --output deployed_mobilepixor.pt
```

---

## 📚 References

- **PIXOR**: Real-time 3D Object Detection from Point Clouds (Yang et al., CVPR 2018)
- **MobileNetV2**: Inverted Residuals and Linear Bottlenecks (Sandler et al., CVPR 2018)
- **PointPillars**: Fast Encoders for Object Detection from Point Clouds (Lang et al., CVPR 2019)
- **CenterNet**: Objects as Points (Zhou et al., arXiv 2019)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
