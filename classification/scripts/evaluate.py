import argparse
import json
import os
import torch
import numpy as np

from classification.models.resnet import ResNet50MC
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.utils.metrics import (
    accuracy, precision, recall, f1, auroc, aupr,
    ece, mce, brier, nll
)
from classification.utils.visualisations import (
    reliability_diagram, predictive_entropy_histogram
)

def evaluate(model, test_loader, device, args):
    """Evaluate a model and return performance metrics."""
    model = model.to(device)
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            all_labels.append(labels)
            all_preds.append(outputs)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    metrics = {
        "accuracy": {
            "macro": float(accuracy(all_preds, all_labels)),
            "per_class": accuracy(all_preds, all_labels, per_class=True)
        },
        "precision": {
            "macro": float(precision(all_preds, all_labels)),
            "per_class": precision(all_preds, all_labels, per_class=True)
        },
        "recall": {
            "macro": float(recall(all_preds, all_labels)),
            "per_class": recall(all_preds, all_labels, per_class=True)
        },
        "f1": {
            "macro": float(f1(all_preds, all_labels)),
            "per_class": f1(all_preds, all_labels, per_class=True)
        },
        "auroc": {
            "macro": float(auroc(all_preds, all_labels)),
            "per_class": auroc(all_preds, all_labels, per_class=True)
        },
        "aupr": {
            "macro": float(aupr(all_preds, all_labels)),
            "per_class": aupr(all_preds, all_labels, per_class=True)
        },
        "ece": float(ece(all_preds, all_labels)),
        "mce": float(mce(all_preds, all_labels)),
        "brier": float(brier(all_preds, all_labels)),
        "nll": float(nll(all_preds, all_labels))
    }

    # --- Save metrics ---
    save_metrics(metrics, args.output_dir)

    # --- Generate plots ---
    reliability_diagram(
        all_preds, all_labels,
        output_path=os.path.join(args.output_dir, "reliability_diagram.png")
    )
    predictive_entropy_histogram(
        all_preds,
        output_path=os.path.join(args.output_dir, "predictive_entropy_histogram.png")
    )

def save_metrics(metrics, output_dir):
    """Save evaluation metrics as a JSON file."""
    def convert(o):
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        return o

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, default=convert)
    print(f"Metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet50 model")

    parser.add_argument("--dataset", type=str, required=True, choices=["aptos2019", "isic2018"], help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save metrics and plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset ---
    if args.dataset.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset.lower() == "isic2018":
        _, _, test_loader, num_classes = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Load model ---
    model = ResNet50MC(num_classes=num_classes, weights=None, dropout_p=args.dropout)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    evaluate(model, test_loader, device=device, args=args)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
