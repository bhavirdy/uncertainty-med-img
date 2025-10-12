import argparse
import json
import os
import torch
import numpy as np

from segmentation.models.unet import UNet, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_segmentation_loaders
from segmentation.utils.segmentation_metrics import (
    dice_score, iou_score, pixel_accuracy, segmentation_ece, 
    segmentation_mce, segmentation_nll, segmentation_brier
)
from segmentation.utils.segmentation_uncertainty import (
    mcdo_segmentation_predictions, segmentation_predictive_mean,
    edl_segmentation_predictions, edl_segmentation_uncertainty
)
from segmentation.utils.segmentation_visualizations import (
    reliability_diagram_from_probs, predictive_entropy_histogram_from_probs,
    uncertainty_heatmap, aleatoric_epistemic_heatmap
)

def evaluate(model, test_loader, device, method, mc_samples):
    model = model.to(device)
    model.eval()

    all_labels, all_probs = [], []
    all_aleatoric, all_epistemic = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if method == "deterministic":
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)

            elif method == "mcdo":
                pred_samples = mcdo_segmentation_predictions(model, imgs, n_samples=mc_samples)
                probs = segmentation_predictive_mean(pred_samples)
                
                # Compute uncertainty measures
                epistemic = segmentation_predictive_variance(pred_samples)
                aleatoric = probs * (1 - probs)
                all_aleatoric.append(aleatoric.cpu())
                all_epistemic.append(epistemic.cpu())

            elif method == "edl":
                alpha = model(imgs)
                probs = alpha / alpha.sum(dim=1, keepdim=True)
                
                # Compute uncertainty measures
                entropy, aleatoric, epistemic = edl_segmentation_uncertainty(alpha)
                all_aleatoric.append(aleatoric.cpu())
                all_epistemic.append(epistemic.cpu())

            else:
                raise ValueError(f"Unknown method {method}")

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    if all_aleatoric:
        all_aleatoric = torch.cat(all_aleatoric)
        all_epistemic = torch.cat(all_epistemic)
    else:
        all_aleatoric = None
        all_epistemic = None

    # --- Performance metrics ---
    metrics = {
        "dice": float(dice_score(all_probs, all_labels)),
        "iou": float(iou_score(all_probs, all_labels)),
        "pixel_accuracy": float(pixel_accuracy(all_probs, all_labels)),
    }

    # --- Uncertainty metrics ---
    metrics.update({
        "ece": float(segmentation_ece(all_probs, all_labels)),
        "mce": float(segmentation_mce(all_probs, all_labels)),
        "nll": float(segmentation_nll(all_probs, all_labels)),
        "brier": float(segmentation_brier(all_probs, all_labels)),
    })

    return metrics, all_probs, all_labels, all_aleatoric, all_epistemic

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
        _, _, test_loader, num_classes = get_isic2018_segmentation_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    # --- Load model ---
    if args.method == "edl":
        model = UNetEDL(n_channels=3, n_classes=num_classes, dropout_p=args.dropout)
    else:
        model = UNet(n_channels=3, n_classes=num_classes, dropout_p=args.dropout)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics, all_probs, all_labels, all_aleatoric, all_epistemic = evaluate(
        model, test_loader, device, method=args.method, mc_samples=args.mc_samples
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
