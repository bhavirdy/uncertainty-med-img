import argparse
import torch
import torch.nn as nn

from models.resnet import get_resnet50
from classification.datasets.aptos2019.aptos_data_loader import get_aptos_loaders
from utils.metrics import (
    accuracy,
    precision_recall_f1,
)
from utils.visualisations import plot_metrics

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Get data loaders based on dataset ---
    if args.dataset.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders()
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Load model ---
    model = get_resnet50(num_classes=num_classes, weights=None)
    model = model.to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc = 0.0, 0.0

    # Lists to store per-batch metrics for plotting
    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            total_acc += accuracy(outputs, labels) * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Average metrics
    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = total_acc / len(test_loader.dataset)
    precision, recall, f1 = precision_recall_f1(all_labels, all_preds, num_classes=num_classes)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Plot metrics
    plot_metrics(
        train_losses=[avg_loss], val_losses=[avg_loss],
        train_accs=[avg_acc], val_accs=[avg_acc],
        train_prec=[precision], val_prec=[precision],
        train_rec=[recall], val_rec=[recall],
        train_f1=[f1], val_f1=[f1],
        dataset_name=args.dataset,
        model_path=args.model_path,
        all_labels=all_labels,
        all_preds=all_preds,
        class_names=list(range(num_classes))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 classifier")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    evaluate(args)
