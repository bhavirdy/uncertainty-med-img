import argparse
import yaml
import json
import os
import torch
import matplotlib as plt

from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.models.resnet import ResNet50MC
from classification.utils.metrics import (
    ece_from_probs, mce_from_probs, brier_from_probs, nll_from_probs
)
from classification.utils.uncertainty import (
    mcdo_predictions,
    predictive_mean
)
from classification.utils.visualisations import (
    reliability_diagram,
    predictive_entropy_histogram,
    confidence_accuracy_plot
)

def mcdo_inference(model, test_loader, device, n_samples=20, output_dir=None):
    all_probs = []
    all_labels = []

    model.to(device)

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # --- MC Dropout samples ---
        pred_samples = mcdo_predictions(model, inputs, n_samples=n_samples)  # [S, B, C]
        probs = predictive_mean(pred_samples)  # [B, C]

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)   # [N, C]
    all_labels = torch.cat(all_labels, dim=0) # [N]

    # --- Metrics ---
    metrics = {
        "ece": ece_from_probs(all_probs, all_labels),
        "mce": mce_from_probs(all_probs, all_labels),
        "brier": brier_from_probs(all_probs, all_labels),
        "nll": nll_from_probs(all_probs, all_labels),
    }

    # --- Plots ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Reliability Diagram
        reliability_diagram(all_probs, all_labels)
        plt.savefig(os.path.join(output_dir, "mcdo_reliability.png"))
        plt.close()

        # Confidence vs Accuracy
        confidence_accuracy_plot(all_probs, all_labels)
        plt.savefig(os.path.join(output_dir, "mcdo_conf_acc.png"))
        plt.close()

        # Predictive Entropy Histogram
        predictive_entropy_histogram(all_probs)
        plt.savefig(os.path.join(output_dir, "mcdo_entropy_hist.png"))
        plt.close()

    return metrics

def deep_ensemble_inference(model, test_loader, device):
    pass

def edl_inference(model, test_loader, device):
    pass

def save_metrics(metrics, output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ue_metrics_all.json"), "w") as f:
            json.dump(metrics, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Uncertainty Estimation Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    model_path = config["model_path"]
    mc_samples = config["mc_samples"]
    output_dir = config["output_dir"]

    if dataset_name.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=batch_size)
    elif dataset_name.lower() == "isic2018":
        _, _, test_loader, num_classes = get_isic2018_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50MC(num_classes=num_classes, dropout_p=dropout).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    metrics = {}

    # Run all methods
    metrics["mcdo"] = mcdo_inference(model=model, test_loader=test_loader, device=device, n_samples=mc_samples, output_dir=output_dir)
    # metrics["deep_ensemble"] = deep_ensemble_inference(model=model, test_loader=test_loader, device=device)
    # metrics["edl"] = edl_inference(model=model, test_loader=test_loader, device=device)

    # --- Save metrics ---
    save_metrics(metrics, output_dir)

if __name__ == "__main__":
    main()