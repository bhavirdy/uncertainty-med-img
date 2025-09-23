import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
import os

class APTOSDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row['diagnosis'])
        return image, label

def get_aptos_loaders(train_csv="./classification/datasets/aptos2019/train_split.csv",
                      val_csv="./classification/datasets/aptos2019/val_split.csv",
                      test_csv="./classification/datasets/aptos2019/test_split.csv",
                      img_dir="./classification/datasets/aptos2019/images",
                      batch_size=32,
                      input_size=224,
                      num_workers=8):

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = APTOSDataset(train_csv, img_dir, transform=train_tf)
    val_set = APTOSDataset(val_csv, img_dir, transform=eval_tf)
    test_set = APTOSDataset(test_csv, img_dir, transform=eval_tf)

    # Weighted sampler for class imbalance
    train_df = pd.read_csv(train_csv)
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_df['diagnosis'].unique())

    return train_loader, val_loader, test_loader, num_classes
