import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class ISIC2018SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # Match images and masks by ID
        self.image_names = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_id = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, img_id + "_segmentation.png")

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to binary (0 = background, 1 = lesion)
        mask = (np.array(mask) > 128).astype(np.uint8)
        mask = Image.fromarray(mask)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = mask.squeeze(0)

        return image, mask

def get_isic2018_segmentation_loaders(
        root="./segmentation/datasets/isic2018seg",
        batch_size=32,
        input_size=224,
        num_workers=8
    ):
    
    train_img_dir = os.path.join(root, "training_input")
    val_img_dir = os.path.join(root, "validation_input")
    test_img_dir = os.path.join(root, "test_input")
    
    train_mask_dir = os.path.join(root, "training_gt")
    val_mask_dir = os.path.join(root, "validation_gt")
    test_mask_dir = os.path.join(root, "test_gt")

    # Transforms for images
    train_img_tf = transforms.Compose([
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

    val_img_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Transforms for masks
    train_mask_tf = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    val_mask_tf = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    # Datasets
    train_set = ISIC2018SegmentationDataset(train_img_dir, train_mask_dir, 
                                           transform=train_img_tf, mask_transform=train_mask_tf)
    val_set = ISIC2018SegmentationDataset(val_img_dir, val_mask_dir, 
                                         transform=val_img_tf, mask_transform=val_mask_tf)
    test_set = ISIC2018SegmentationDataset(test_img_dir, test_mask_dir, 
                                          transform=val_img_tf, mask_transform=val_mask_tf)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = 2  # Background and lesion

    return train_loader, val_loader, test_loader, num_classes
