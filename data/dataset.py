from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset


class SplitImageDataset(Dataset):
    """Load RGB images referenced by a generated split manifest."""

    def __init__(self, rows, data_root, transform):
        self.rows = rows
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["file_name"]
        with Image.open(path) as img:
            image = self.transform(ImageOps.exif_transpose(img).convert("RGB"))
        return image, int(row["class_index"]), row["file_name"]
