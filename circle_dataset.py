from pathlib import Path
from typing import List, Tuple
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CircleDataset(Dataset):
    """Dataset for training circle detection in YOLO format."""

    def __init__(self, data_yaml: str, split: str = "train", img_size: int = 640):
        cfg = yaml.safe_load(open(data_yaml, 'r'))
        path = Path(cfg.get('path', '.'))
        self.img_dir = path / cfg.get(f'{split}', 'images/' + split)
        self.label_dir = path / 'labels' / split
        self.img_files = sorted(self.img_dir.glob('*.*'))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        targets = []
        if label_path.exists():
            for line in open(label_path):
                cls, x, y, r = map(float, line.split())
                targets.append([cls, x, y, r])
        targets = torch.tensor(targets, dtype=torch.float32)
        return img, targets

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        imgs, targets = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, targets
