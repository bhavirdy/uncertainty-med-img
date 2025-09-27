import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import os

class ISIC2018Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Label columns
        self.label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"] + ".jpg")

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Multi-class one-hot label
        label = torch.tensor(row[self.label_cols].values.astype("float32"))

        return image, label

def get_isic2018_loaders(
        root="./classification/datasets/isic2018",
        batch_size=32,
        input_size=224,
        num_workers=8
    ):
    train_csv = os.path.join(root, "training_gt", "training_gt.csv")
    val_csv = os.path.join(root, "validation_gt", "validation_gt.csv")
    test_csv = os.path.join(root, "test_gt", "test_gt.csv")

    train_dir = os.path.join(root, "training_input")
    val_dir = os.path.join(root, "validation_input")
    test_dir = os.path.join(root, "test_input")

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_set = ISIC2018Dataset(train_csv, train_dir, transform=train_tf)
    val_set = ISIC2018Dataset(val_csv, val_dir, transform=val_tf)
    test_set = ISIC2018Dataset(test_csv, test_dir, transform=val_tf)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_set.label_cols)

    return train_loader, val_loader, test_loader, num_classes
