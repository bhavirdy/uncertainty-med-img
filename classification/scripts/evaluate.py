import argparse
import yaml
import json
import os
import torch
import numpy as np

from classification.models.resnet import ResNet50MC
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.utils.metrics import accuracy, precision, recall, f1

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

    # Concatenate all batches
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Compute metrics
    acc = accuracy(all_preds, all_labels)
    prec = precision(all_preds, all_labels)
    rec = recall(all_preds, all_labels)
    f1_score_val = f1(all_preds, all_labels)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_score_val)
    }

    return metrics

def save_metrics(metrics, output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
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

    if dataset_name.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    model = ResNet50MC(num_classes=num_classes, weights=None, dropout_p=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics = evaluate(model, test_loader, device=device)

    # --- Print results ---
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1']:.4f}")

    # --- Save metrics ---
    save_metrics(metrics, output_dir)
