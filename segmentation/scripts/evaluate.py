import argparse
import json
import os
import torch
from torchmetrics import JaccardIndex, F1Score, Accuracy, AUROC, AveragePrecision, CalibrationError

from segmentation.models.unet import UNet, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_loaders
from segmentation.utils.metrics import nll, brier
from segmentation.utils.uncertainty import mcdo_predictions, predictive_mean, predictive_variance
from segmentation.utils.visualizations import (
    reliability_diagram_from_probs, predictive_entropy_histogram_from_probs,
    uncertainty_heatmap, aleatoric_epistemic_heatmap
)

def evaluate(model, test_loader, device, args):
    model = model.to(device)
    model.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if args.method == "deterministic":
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)

            elif args.method == "mcdo":
                pred_samples = mcdo_predictions(model, imgs, n_samples=args.mc_samples)
                probs = predictive_mean(pred_samples)

            elif args.method == "edl":
                alpha = model(imgs)
                probs = alpha / alpha.sum(dim=1, keepdim=True)

            # --- Collect all predictions and labels ---
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # --- Concatenate all batches ---
    all_labels = torch.cat(all_labels)           # [B, H, W]
    all_probs = torch.cat(all_probs)             # [B, C, H, W]

    # --- Flatten for per-pixel metrics ---
    B, C, H, W = all_probs.shape
    all_probs_flat = all_probs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    all_labels_flat = all_labels.reshape(-1)                        # [B*H*W]
    preds_flat = torch.argmax(all_probs_flat, dim=1)

    # --- Compute metrics ---
    dice = F1Score(num_classes=args.num_classes, average='macro')(preds_flat, all_labels_flat).item()
    iou = JaccardIndex(num_classes=args.num_classes)(preds_flat, all_labels_flat).item()
    acc = Accuracy(num_classes=args.num_classes)(preds_flat, all_labels_flat).item()
    ece = CalibrationError(n_bins=15, norm='l1', num_classes=args.num_classes)(all_probs_flat, all_labels_flat).item()
    mce = CalibrationError(n_bins=15, norm='max', num_classes=args.num_classes)(all_probs_flat, all_labels_flat).item()
    auroc = AUROC(num_classes=args.num_classes, average='macro')(all_probs_flat, all_labels_flat).item()
    aupr = AveragePrecision(num_classes=args.num_classes, average='macro')(all_probs_flat, all_labels_flat).item()

    # --- Store metrics ---
    metrics = {
        "dice": dice,
        "iou": iou,
        "pixel_accuracy": acc,
        "auroc": auroc,
        "aupr": aupr,
        "ece": ece,
        "mce": mce,
        "nll": float(nll(all_probs_flat, all_labels_flat)),
        "brier": float(brier(all_probs_flat, all_labels_flat)),
    }

    return metrics, all_probs, all_labels

def save_metrics(metrics, output_dir, method):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{method}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

def generate_plots(all_probs, all_labels, output_dir, method, all_aleatoric=None, all_epistemic=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Reliability diagram
    reliability_diagram_from_probs(all_probs, all_labels,
                                  output_path=os.path.join(output_dir, f"{method}_reliability_diagram.png"))
    
    # Entropy histogram
    predictive_entropy_histogram_from_probs(all_probs, all_labels,
                                           output_path=os.path.join(output_dir, f"{method}_entropy_histogram.png"))
    
    # Uncertainty heatmap
    uncertainty_heatmap(all_probs, all_labels,
                       output_path=os.path.join(output_dir, f"{method}_uncertainty_heatmap.png"),
                       method='entropy')
    
    # Aleatoric vs Epistemic uncertainty (if available)
    if all_aleatoric is not None and all_epistemic is not None:
        aleatoric_epistemic_heatmap(all_aleatoric, all_epistemic, all_labels,
                                   output_path=os.path.join(output_dir, f"{method}_aleatoric_epistemic_heatmap.png"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model with uncertainty metrics")

    parser.add_argument("--dataset", type=str, required=True, choices=["isic2018"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["deterministic", "mcdo", "edl"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mc_samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset ---
    if args.dataset.lower() == "isic2018":
        _, _, test_loader, num_classes = get_isic2018_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    args.num_classes = num_classes

    # --- Load model ---
    if args.method == "edl":
        model = UNetEDL(n_channels=3, n_classes=num_classes, dropout_p=args.dropout)
    else:
        model = UNet(n_channels=3, n_classes=num_classes, dropout_p=args.dropout)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics, all_probs, all_labels, all_aleatoric, all_epistemic = evaluate(
        model, test_loader, device, method=args.method, mc_samples=args.mc_samples, args=args
    )
    
    save_metrics(metrics, args.output_dir, args.method)
    generate_plots(all_probs, all_labels, args.output_dir, args.method, all_aleatoric, all_epistemic)
    
    # Print metrics
    print(f"\nEvaluation Results for {args.method}:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
