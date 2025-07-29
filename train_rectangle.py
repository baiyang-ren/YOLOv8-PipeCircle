import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 rectangle model")

    parser.add_argument('--data', type=str, default='datasets/joint_any_v2_full/data_AnyJoint.yaml', help='Dataset YAML')

    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda', help='Computation device')
    return parser.parse_args()


def main():
    args = parse_args()
    from torch.utils.data import DataLoader
    import torch
    from rectangle_model import load_yolov8_rectangle
    from rectangle_dataset import RectangleDataset
    from rectangle_loss import RectangleLoss

    print(f"Loading dataset from: {args.data}")
    cfg = yaml.safe_load(open(args.data))
    nc = cfg['nc']
    print(f"Number of classes: {nc}")
    
    print(f"Loading model with weights: {args.weights}")
    model = load_yolov8_rectangle(args.weights, nc)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.model.to(device)
    
    print("Creating dataset and dataloader...")
    dataset = RectangleDataset(args.data, 'train', img_size=args.img_size)
    print(f"Dataset size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=RectangleDataset.collate_fn)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    criterion = RectangleLoss()

    print("Starting training...")
    model.model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (imgs, targets) in enumerate(loader):
            try:
                imgs = imgs.to(device)
                preds = model.model(imgs)
                if isinstance(preds, tuple):
                    preds = preds[0]
                elif isinstance(preds, list):
                    preds = preds[0]  # Take the first element if it's a list
                loss = criterion(preds, targets, nc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch+1}/{args.epochs} average loss: {avg_loss:.4f}")

    print("Training completed. Saving model...")
    model.model.eval()
    model.save('rectangle_trained.pt')
    print("Model saved as 'rectangle_trained.pt'")


if __name__ == '__main__':
    main() 