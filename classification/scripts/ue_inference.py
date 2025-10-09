import argparse
import json
import os
import torch

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
    reliability_diagram_from_probs,
    predictive_entropy_histogram_from_probs
)

def mcdo_inference(model, test_loader, device, args):
    """Perform Monte Carlo Dropout inference and compute uncertainty metrics."""
    all_probs = []
    all_labels = []

    model.to(device)
    model.eval()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # --- Monte Carlo Dropout samples ---
        pred_samples = mcdo_predictions(model, inputs, n_samples=args.mc_samples)  # [S, B, C]
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
    os.makedirs(args.output_dir, exist_ok=True)

    reliability_path = os.path.join(args.output_dir, "mcdo_reliability.png")
    reliability_diagram_from_probs(all_probs, all_labels, output_path=reliability_path)

    entropy_path = os.path.join(args.output_dir, "mcdo_entropy_hist.png")
    predictive_entropy_histogram_from_probs(all_probs, output_path=entropy_path)

    return metrics

def save_metrics(metrics, output_dir):
    """Save computed metrics as JSON."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "ue_metrics_all.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="Uncertainty Estimation Inference")

    parser.add_argument("--dataset", type=str, required=True, choices=["aptos2019", "isic2018"], help="Dataset name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save metrics and plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--mc_samples", type=int, default=20, help="Number of Monte Carlo Dropout samples")
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
    model = ResNet50MC(num_classes=num_classes, dropout_p=args.dropout).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Run MC Dropout inference ---
    metrics = {}
    metrics["mcdo"] = mcdo_inference(model, test_loader, device, args)

    save_metrics(metrics, args.output_dir)
    print("Uncertainty estimation completed successfully.")

if __name__ == "__main__":
    main()
