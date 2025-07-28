import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLOv8 model")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Image or directory to run inference on')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--save', action='store_true', help='Save predictions to file')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device')
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=args.img_size,
                           save=args.save, device=args.device)
    for r in results:
        boxes = r.boxes
        print(boxes)


if __name__ == '__main__':
    main()
