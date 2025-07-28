# YOLOv8-PipeCircle

This repository provides a basic setup for training and running inference with a YOLOv8 object detection model using PyTorch. Pretrained weights are downloaded automatically by the [Ultralytics](https://github.com/ultralytics/ultralytics) package.

## Setup

Install PyTorch with GPU support following the [official instructions](https://pytorch.org/get-started/locally/)
for your CUDA version. After PyTorch is installed, install the remaining
dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place your dataset in the `datasets/` directory using the standard YOLO format and create a YAML configuration file describing the dataset (see Ultralytics documentation for details).

## Training

### Bounding Box Model

Run training with:

```bash
python train.py --data datasets/your_dataset.yaml --weights yolov8n.pt --epochs 100
```
For example, to train on the first GPU:
```bash
python train.py --data datasets/your_dataset.yaml --device 0
```

The training script detects a CUDA capable device automatically. Use the
`--device` option to explicitly choose a device (e.g. `0` for the first GPU or
`cpu`).

Use the `--freeze` argument to freeze the first N layers of the network when fine‑tuning.

### Bounding Circle Model

Phase 2 introduces a modified YOLO head that predicts circle center `(x, y)` and radius `r` instead of a bounding box. Train this model with:

```bash
python train_circle.py --data datasets/your_dataset.yaml --weights yolov8n.pt --epochs 100
```

As with the box model, training uses the GPU when available. Specify a
particular device with `--device` if needed.

## Inference

After training, run inference on images with:

```bash
python infer.py --weights path/to/best.pt --source path/to/images --save
```

Inference also defaults to the first CUDA device when available. Pass
`--device cpu` to force CPU mode.
Example using a GPU:
```bash
python infer.py --weights path/to/best.pt --source images/ --device 0
```

Results will be printed to the console and optionally saved alongside the source images.

For the circle model use:

```bash
python infer_circle.py --weights circle_trained.pt --data datasets/your_dataset.yaml --source path/to/image.jpg --save
```

The `--device` option is available here as well.
Example:
```bash
python infer_circle.py --weights circle_trained.pt --data datasets/your_dataset.yaml --source img.jpg --device 0
```

