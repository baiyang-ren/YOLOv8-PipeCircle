import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset")
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--freeze', type=int, default=None, help='Number of layers to freeze')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device')
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.img_size,
                batch=args.batch, freeze=args.freeze, device=args.device)


if __name__ == '__main__':
    main()
