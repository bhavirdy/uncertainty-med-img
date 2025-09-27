import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop_v2(img_path):
    """
    Create circular crop around image centre.
    Returns a numpy image (H, W, C) in RGB order.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    x, y = int(width / 2), int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

class APTOSDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None, circle_crop=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.circle_crop = circle_crop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + ".png")

        # Apply circle crop preprocessing
        if self.circle_crop:
            image = circle_crop_v2(img_path)
            image = Image.fromarray(image)  # Convert numpy → PIL for torchvision transforms
        else:
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
                      num_workers=8,
                      circle_crop=True):

    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = APTOSDataset(train_csv, img_dir, transform=tf, circle_crop=circle_crop)
    val_set = APTOSDataset(val_csv, img_dir, transform=tf, circle_crop=circle_crop)
    test_set = APTOSDataset(test_csv, img_dir, transform=tf, circle_crop=circle_crop)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_df = pd.read_csv(train_csv)
    num_classes = len(train_df['diagnosis'].unique())

    return train_loader, val_loader, test_loader, num_classes
