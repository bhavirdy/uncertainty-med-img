import argparse
import json
import os
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, CalibrationError

from classification.models.resnet import ResNet50, ResNet50EDL
from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.data_loaders.isic2018_data_loader import get_isic2018_loaders
from classification.utils.metrics import brier, nll
from classification.utils.uncertainty import mcdo_predictions, predictive_mean
from classification.utils.visualisations import reliability_diagram_from_probs, predictive_entropy_histogram_from_probs

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
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    preds = torch.argmax(all_probs, dim=1)

    # --- Compute metrics ---
    acc = Accuracy(num_classes=args.num_classes)(preds, all_labels).item()
    precision = Precision(num_classes=args.num_classes, average='macro')(preds, all_labels).item()
    recall = Recall(num_classes=args.num_classes, average='macro')(preds, all_labels).item()
    f1 = F1Score(num_classes=args.num_classes, average='macro')(preds, all_labels).item()
    auroc = AUROC(num_classes=args.num_classes, average='macro')(all_probs, all_labels).item()
    aupr = AveragePrecision(num_classes=args.num_classes, average='macro')(all_probs, all_labels).item()
    ece = CalibrationError(n_bins=15, norm='l1', num_classes=args.num_classes)(all_probs, all_labels).item()
    mce = CalibrationError(n_bins=15, norm='max', num_classes=args.num_classes)(all_probs, all_labels).item()

    # --- Store metrics ---
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "aupr": aupr,
        "ece": ece,
        "mce": mce,
        "brier": brier(all_probs, all_labels),
        "nll": nll(all_probs, all_labels)
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
    reliability_diagram_from_probs(all_probs, all_labels, output_path=os.path.join(output_dir, f"{method}_reliability.png"))
    predictive_entropy_histogram_from_probs(all_probs, output_path=os.path.join(output_dir, f"{method}_entropy_hist.png"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model with uncertainty metrics")

    parser.add_argument("--dataset", type=str, required=True, choices=["aptos2019", "isic2018"])
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset')
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
        _, _, test_loader = get_aptos_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset.lower() == "isic2018":
        _, _, test_loader = get_isic2018_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # --- Load model ---
    if args.method == "edl":
        model = ResNet50EDL(num_classes=args.num_classes, dropout_p=args.dropout)
    else:
        model = ResNet50(num_classes=args.num_classes, dropout_p=args.dropout)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    # --- Evaluate ---
    metrics, all_probs, all_labels = evaluate(model, test_loader, device, args)
    save_metrics(metrics, args.output_dir, args.method)
    generate_plots(all_probs, all_labels, args.output_dir, args.method)

if __name__ == "__main__":
    main()
