import argparse
import yaml
from torch.utils.data import DataLoader
import torch
from circle_model import load_yolov8_circle
from circle_dataset import CircleDataset
from circle_loss import CircleLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 circle model")
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.data))
    nc = cfg['nc']
    model = load_yolov8_circle(args.weights, nc)
    dataset = CircleDataset(args.data, 'train', img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=CircleDataset.collate_fn)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    criterion = CircleLoss()

    model.model.train()
    for epoch in range(args.epochs):
        for imgs, targets in loader:
            preds = model.model(imgs)[0]
            loss = criterion(preds, targets, nc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs} loss: {loss.item():.4f}")

    model.model.eval()
    model.model.save('circle_trained.pt')


if __name__ == '__main__':
    main()
