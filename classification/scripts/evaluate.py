import argparse
import json
import os
import torch

from classification.models.resnet import ResNet50
from classification.models.resnet_edl import ResNet50EDL
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.utils.metrics import (
    accuracy, precision, recall, f1, auroc, aupr,
    ece, mce, brier, nll
)
from classification.utils.uncertainty import mcdo_predictions, predictive_mean
from classification.utils.visualisations import (
    reliability_diagram_from_probs, predictive_entropy_histogram_from_probs
)

def evaluate(model, test_loader, device, method, mc_samples):
    model = model.to(device)
    model.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if method == "deterministic":
                probs = torch.softmax(model(imgs), dim=1)

            elif method == "mcdo":
                pred_samples = mcdo_predictions(model, imgs, n_samples=mc_samples)
                probs = predictive_mean(pred_samples)

            elif method == "edl":
                alpha = model(imgs)
                probs = alpha / alpha.sum(dim=1, keepdim=True)

            else:
                raise ValueError(f"Unknown method {method}")

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # --- Standard performance metrics ---
    metrics = {
        "accuracy": float(accuracy(all_probs, all_labels)),
        "precision": float(precision(all_probs, all_labels)),
        "recall": float(recall(all_probs, all_labels)),
        "f1": float(f1(all_probs, all_labels)),
        "auroc": float(auroc(all_probs, all_labels)),
        "aupr": float(aupr(all_probs, all_labels)),
    }

    # --- Uncertainty metrics ---
    metrics.update({
        "ece": ece(all_probs, all_labels),
        "mce": mce(all_probs, all_labels),
        "brier": brier(all_probs, all_labels),
        "nll": nll(all_probs, all_labels)
    })

    return metrics, all_probs, all_labels

def save_metrics(metrics, output_dir, method):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_{method}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {method} saved to {path}")

def generate_plots(all_probs, all_labels, output_dir, method):
    os.makedirs(output_dir, exist_ok=True)
    reliability_diagram_from_probs(all_probs, all_labels,
                                   output_path=os.path.join(output_dir, f"{method}_reliability.png"))
    predictive_entropy_histogram_from_probs(all_probs,
                                            output_path=os.path.join(output_dir, f"{method}_entropy_hist.png"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model with uncertainty metrics")

    parser.add_argument("--dataset", type=str, required=True, choices=["aptos2019", "isic2018"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["deterministic", "mcdo", "edl"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mc_samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset ---
    if args.dataset.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset.lower() == "isic2018":
        _, _, test_loader, num_classes = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Load model ---
    if args.method == "edl":
        model = ResNet50EDL(num_classes=num_classes, dropout_p=args.dropout)
    else:
        model = ResNet50(num_classes=num_classes, dropout_p=args.dropout)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics, all_probs, all_labels = evaluate(model, test_loader, device, method=args.method, mc_samples=args.mc_samples)
    save_metrics(metrics, args.output_dir, args.method)
    generate_plots(all_probs, all_labels, args.output_dir, args.method)

if __name__ == "__main__":
    main()
