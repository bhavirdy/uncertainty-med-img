import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISIC2018SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Match images and masks by ID
        self.image_names = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir) if f.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_id = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, img_id + "_segmentation.png")

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.uint8)  # binary mask

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

def get_transforms(input_size=224):
    train_transform = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.RandomResizedCrop(size=(input_size, input_size), scale=(0.9,1.0), p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

    val_transform = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

    return train_transform, val_transform

def get_isic2018_loaders(
        root="./segmentation/datasets/isic2018seg",
        batch_size=16,
        input_size=224,
        num_workers=8
    ):
    train_img_dir = os.path.join(root, "training_input")
    val_img_dir = os.path.join(root, "validation_input")
    test_img_dir = os.path.join(root, "test_input")

    train_mask_dir = os.path.join(root, "training_gt")
    val_mask_dir = os.path.join(root, "validation_gt")
    test_mask_dir = os.path.join(root, "test_gt")

    train_tf, val_tf = get_transforms(input_size)

    train_set = ISIC2018SegmentationDataset(train_img_dir, train_mask_dir, transform=train_tf)
    val_set = ISIC2018SegmentationDataset(val_img_dir, val_mask_dir, transform=val_tf)
    test_set = ISIC2018SegmentationDataset(test_img_dir, test_mask_dir, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = 2  # Background + lesion

    return train_loader, val_loader, test_loader, num_classes

def get_isic2018_loaders_test(
        root="./segmentation/datasets/isic2018seg",
        batch_size=8,
        input_size=224,
        num_workers=4
    ):
    train_img_dir = os.path.join(root, "validation_input")
    train_mask_dir = os.path.join(root, "validation_gt")

    train_tf, _ = get_transforms(input_size)

    train_set = ISIC2018SegmentationDataset(train_img_dir, train_mask_dir, transform=train_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    num_classes = 2  # Background + lesion

    return train_loader, num_classes