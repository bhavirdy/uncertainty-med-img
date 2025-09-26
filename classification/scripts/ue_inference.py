import argparse
import yaml
import json
import os
import torch

from classification.data_loaders.aptos_data_loader import get_aptos_loaders
from classification.models.resnet import ResNet50MC
from classification.utils.uncertainty import (
    mcdo_predictions,
    deep_ensemble_predictions,
    edl_predictions,
    predictive_mean,
    predictive_entropy
)
from classification.utils.metrics import (
    accuracy, precision, recall, f1,
    ece, brier, nll, auroc, aupr
)

def mcdo_inference(config, device):
    print("Running MC Dropout inference...")
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    model_path = config["model_path"]
    mc_samples = config["mc_samples"]

    if dataset_name.lower() == "aptos2019":
        _, _, test_loader, num_classes = get_aptos_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    model = ResNet50MC(num_classes=num_classes, dropout_p=dropout).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    pred_samples, labels = mcdo_predictions(model, test_loader, device, n_samples=mc_samples)
    return compute_metrics(pred_samples, labels)

def deep_ensemble_inference(config, device):
    print("Running Deep Ensemble inference...")
    pass

def edl_inference(config, device):
    print("Running Evidential Deep Learning inference...")
    pass

def compute_metrics(pred_samples, labels):
    mean_probs = predictive_mean(pred_samples)

    metrics = {
        "accuracy": accuracy(mean_probs, labels),
        "precision": precision(mean_probs, labels),
        "recall": recall(mean_probs, labels),
        "f1": f1(mean_probs, labels),
        "ece": ece(mean_probs, labels),
        "brier": brier(mean_probs, labels),
        "nll": nll(mean_probs, labels),
        "auroc": None,  # TODO
        "aupr": None,   # TODO
        "aurc": None,   # TODO
        "fpr@95": None  # TODO
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty Estimation Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config["output_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # Run all methods
    results["mcdo"] = mcdo_inference(config, device)
    # results["deep_ensemble"] = deep_ensemble_inference(config, device)
    # results["edl"] = edl_inference(config, device)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ue_metrics_all.json"), "w") as f:
            json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))
