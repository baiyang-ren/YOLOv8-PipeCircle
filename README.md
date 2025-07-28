# YOLOv8-PipeCircle

This repository provides a basic setup for training and running inference with a YOLOv8 object detection model using PyTorch. Pretrained weights are downloaded automatically by the [Ultralytics](https://github.com/ultralytics/ultralytics) package.

## Setup

Install dependencies using `pip`:

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

Use the `--freeze` argument to freeze the first N layers of the network when fine‑tuning.

### Bounding Circle Model

Phase 2 introduces a modified YOLO head that predicts circle center `(x, y)` and radius `r` instead of a bounding box. Train this model with:

```bash
python train_circle.py --data datasets/your_dataset.yaml --weights yolov8n.pt --epochs 100
```

## Inference

After training, run inference on images with:

```bash
python infer.py --weights path/to/best.pt --source path/to/images --save
```

Results will be printed to the console and optionally saved alongside the source images.

For the circle model use:

```bash
python infer_circle.py --weights circle_trained.pt --data datasets/your_dataset.yaml --source path/to/image.jpg --save
```

