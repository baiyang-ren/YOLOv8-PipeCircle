import argparse
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with circle detection model")
    parser.add_argument('--weights', type=str, required=True, help='Trained weights')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML for class names')
    parser.add_argument('--source', type=str, required=True, help='Image path')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device')
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics.utils.plotting import Annotator
    from circle_model import load_yolov8_circle
    from PIL import Image
    import torch

    cfg = yaml.safe_load(open(args.data))
    nc = cfg['nc']
    names = cfg.get('names', list(range(nc)))
    model = load_yolov8_circle(args.weights, nc)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model.model.to(device)
    img = Image.open(args.source).convert('RGB')
    img_resized = img.resize((args.img_size, args.img_size))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    pred = model.model(img_tensor.unsqueeze(0).to(device))[0]
    annotator = Annotator(img)
    for c, score, cls in pred:
        x, y, r = c
        annotator.circle((float(x), float(y)), float(r), label=names[int(cls)], color=(255,0,0))
    if args.save:
        out = args.source.replace('.', '_pred.')
        annotator.result().save(out)
        print(f"Saved to {out}")
    else:
        annotator.result().show()


if __name__ == '__main__':
    main()
