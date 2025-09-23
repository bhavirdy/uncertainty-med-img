import argparse
import yaml
import json
import os
import torch
import numpy as np

from classification.models.resnet import get_resnet50
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.utils.metrics import precision_recall_f1
from classification.utils.visualisations import plot_confusion_matrix

def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Get data loaders based on dataset ---
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    output_dir = config["output_dir"]
    model_path = config["model_path"]
    save_preds = config["save_preds"]

    if dataset_name.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # --- Load model ---
    model = get_resnet50(num_classes=num_classes, weights=None)
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Metrics
    avg_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    precision, recall, f1 = precision_recall_f1(all_labels, all_preds)

    print(f"Test Accuracy: {avg_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Save metrics to JSON
    metrics = {
        "accuracy": float(avg_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # Save raw predictions
    if save_preds and output_dir:
        preds_path = os.path.join(output_dir, "preds.npy")
        np.save(preds_path, np.array(all_preds))

    # Plot and save confusion matrix
    plot_confusion_matrix(
        all_labels=all_labels,
        all_preds=all_preds,
        model_path=model_path,
        class_names=list(range(num_classes)),
        output_dir=output_dir
    )

    # --- Save a copy of config ---
    config_copy_path = os.path.join(output_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.safe_dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config)