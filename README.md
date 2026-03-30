# AnyLaneNET

Multi-lane type detector – from sidewalks to hallways – for autonomous rover.
Built as an encoder-decoder CNN (ResNet-18 backbone) trained for binary lane segmentation.

---

## Project layout

```
AnyLaneNET/
├── configs/
│   └── default.yaml          # Training hyper-parameters and paths
├── data/
│   ├── raw/
│   │   ├── images/           # Input RGB frames  (*.png)
│   │   └── masks/            # Binary lane masks (*.png, 0/255)
│   └── processed/            # (optional) pre-processed artefacts
├── checkpoints/              # Saved model weights (.pth)
├── logs/                     # TensorBoard event files
├── notebooks/
│   └── explore_data.ipynb    # Data / augmentation exploration
├── src/
│   ├── anylane/
│   │   ├── models/
│   │   │   └── lane_net.py   # LaneNet architecture
│   │   ├── data/
│   │   │   ├── dataset.py    # LaneDataset (PyTorch Dataset)
│   │   │   └── transforms.py # Albumentations pipelines
│   │   └── utils/
│   │       ├── metrics.py    # IoU, accuracy
│   │       └── visualization.py
│   ├── train.py              # Training entry-point
│   └── evaluate.py           # Evaluation / inference entry-point
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_metrics.py
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies (GPU / Linux with CUDA 12.1)

```bash
pip install -r requirements.txt
```

> **Different CUDA version?** Edit the `--extra-index-url` line in `requirements.txt` to match
> your CUDA version (e.g. `cu118` for CUDA 11.8).

### 3. Install the package in editable mode

```bash
pip install -e ".[dev]"
```

---

## Usage

### 1. Generate training data

Extract RGB frames and binary lane masks from a video:

```bash
python src/datagen.py path/to/video.MOV
```

Add `--preview` to see the overlay as it generates:

```bash
python src/datagen.py path/to/video.MOV --preview
```

This saves paired images and masks to `data/raw/images/` and `data/raw/masks/`.

### 2. Train

```bash
python src/train.py --config configs/default.yaml
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir logs
```

### 3. Inference

Run the trained model on a video:

```bash
python src/inference.py path/to/video.MOV
```

Adjust detection confidence with `--threshold` (default 0.5):

```bash
python src/inference.py path/to/video.MOV --threshold 0.3
```

Press `q` to quit the preview window.

### 4. Export to ONNX

Export the trained model for deployment (e.g. TensorRT on Jetson):

```bash
python src/export.py
```

This produces `lanenet.onnx`. To build a TensorRT engine on the target device:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=lanenet.onnx --saveEngine=lanenet.engine --fp16
```

# Release to repo
`gh release create v1.0 lanenet.onnx --title "v1.0" --notes "Initial lane detection model"`

---

## Evaluate

```bash
python src/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth
```

---

## Test

```bash
pytest
```

---

## Configuration

All training settings are in `configs/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model.num_classes` | `1` | Output channels (1 = binary) |
| `model.pretrained` | `true` | ImageNet encoder weights |
| `data.img_height/width` | `360 / 640` | Resize target |
| `data.val_split` | `0.15` | Validation fraction |
| `training.epochs` | `50` | Training epochs |
| `training.batch_size` | `8` | Batch size |
| `training.lr` | `1e-4` | Initial learning rate |
