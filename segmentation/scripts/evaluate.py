import argparse
import json
import os
import torch
from torchmetrics import JaccardIndex, F1Score, CalibrationError
from sklearn.metrics import brier_score_loss, log_loss

from segmentation.models.unet import UNetDeterministic, UNetMCDO, UNetEDL
from segmentation.data_loaders.isic2018_segmentation_data_loader import get_isic2018_loaders
from segmentation.utils.uncertainty import mcdo_predictions, predictive_mean
from segmentation.utils.visualisations import plot_reliability_diagram, plot_entropy_heatmaps

def evaluate(model, test_loader, device, args):
    model = model.to(device)
    model.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            if args.method == "deterministic":
                probs = torch.softmax(outputs, dim=1)

            elif args.method == "mcdo":
                pred_samples = mcdo_predictions(model, imgs, n_samples=args.mc_samples)
                probs = predictive_mean(pred_samples)

            elif args.method == "edl":
                alpha = outputs
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
    dice = F1Score(task='multiclass', num_classes=args.num_classes, average='macro')(preds_flat, all_labels_flat).item()
    iou = JaccardIndex(task='multiclass', num_classes=args.num_classes, average='macro')(preds_flat, all_labels_flat).item()

    ece = CalibrationError(task='multiclass', n_bins=15, norm='l1', num_classes=args.num_classes)(all_probs_flat, all_labels_flat).item()
    mce = CalibrationError(task='multiclass', n_bins=15, norm='max', num_classes=args.num_classes)(all_probs_flat, all_labels_flat).item()
    brier = brier_score_loss(all_labels_flat.numpy(), all_probs_flat.numpy())
    nll = log_loss(all_labels_flat.numpy(), all_probs_flat.numpy())

    # --- Store metrics ---
    metrics = {
        "dice": dice,
        "iou": iou,
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "brier": brier
    }

    return metrics, all_probs, all_labels

def save_metrics(metrics, output_dir, method):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_{method}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {method} saved to {path}")

def generate_plots(all_probs, all_labels, output_dir, method):
    os.makedirs(output_dir, exist_ok=True)
    plot_reliability_diagram(all_probs, all_labels, output_path=os.path.join(output_dir, f"{method}_reliability_diagram.png"))
    plot_entropy_heatmaps(all_probs, all_labels, output_path=os.path.join(output_dir, f"{method}_entropy_heatmaps.png"))    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, choices=["isic2018"])
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    
    parser.add_argument("--method", type=str, required=True, choices=["deterministic", "mcdo", "edl"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mc_samples", type=int, default=20)
    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset ---
    if args.dataset.lower() == "isic2018":
        _, _, test_loader = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # --- Model ---
    if args.method == "deterministic":
        model = UNetDeterministic(n_classes=args.num_classes)
    elif args.method == "mcdo":
        model = UNetMCDO(n_classes=args.num_classes, dropout_p=args.dropout)
    elif args.method == "edl":
        model = UNetEDL(n_classes=args.num_classes)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics, all_probs, all_labels = evaluate(model, test_loader, device, args)
    save_metrics(metrics, args.output_dir, args.method)
    generate_plots(all_probs, all_labels, args.output_dir, args.method)

if __name__ == "__main__":
    main()
