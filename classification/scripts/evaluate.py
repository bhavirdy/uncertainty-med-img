import argparse
import yaml
import json
import os
import torch
import numpy as np

from classification.models.resnet import ResNet50MC
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.utils.metrics import accuracy, precision, recall, f1, auroc, aupr, ece, mce, brier, nll
from classification.utils.visualisations import reliability_diagram, predictive_entropy_histogram

def evaluate(model, test_loader, device):
    """
    Evaluate a model and return metrics.
    """
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
    }

    # --- Uncertainty metrics ---
    metrics["ece"] = float(ece(all_preds, all_labels))
    metrics["mce"] = float(mce(all_preds, all_labels))
    metrics["brier"] = float(brier(all_preds, all_labels))
    metrics["nll"] = float(nll(all_preds, all_labels))

    return metrics, all_preds, all_labels

def save_metrics(metrics, output_dir):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return o

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, default=convert)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Prepare model and dataloader ---
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    model_path = config["model_path"]
    output_dir = config["output_dir"]
    num_workers = config["num_workers"]

    if dataset_name.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=batch_size, num_workers=num_workers)
    elif dataset_name.lower() == "isic2018":
        _, _, test_loader, num_classes = get_isic2018_loaders(batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    model = ResNet50MC(num_classes=num_classes, weights=None, dropout_p=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics, all_preds, all_labels = evaluate(model, test_loader, device=device)

    # --- Save metrics ---
    save_metrics(metrics, output_dir)

    # --- Generate plots ---
    reliability_diagram(
        all_preds, all_labels,
        output_path=os.path.join(output_dir, "reliability_diagram.png")
    )
    predictive_entropy_histogram(
        all_preds,
        output_path=os.path.join(output_dir, "predictive_entropy_histogram.png")
    )

if __name__ == "__main__":
    main()