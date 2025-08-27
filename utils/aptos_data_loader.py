import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------
# Dataset class
# ----------------------------
class APTOSDataset(Dataset):
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

# ----------------------------
# Function to create DataLoaders
# ----------------------------
def get_aptos_loaders(train_csv="datasets/aptos2019/train_split.csv", val_csv="datasets/aptos2019/val_split.csv", test_csv="datasets/aptos2019/test_split.csv",
                      img_dir="datasets/aptos2019/images",
                      batch_size=32,
                      input_size=224,
                      num_workers=4):

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_set = APTOSDataset(train_csv, img_dir, transform=train_tf)
    val_set = APTOSDataset(val_csv, img_dir, transform=eval_tf)
    test_set = APTOSDataset(test_csv, img_dir, transform=eval_tf)

    # Number of classes
    num_classes = len(pd.read_csv(train_csv)['diagnosis'].unique())

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, num_classes
