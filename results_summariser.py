import json
import os
import numpy as np

tasks = ["classification", "segmentation"]
methods = ["deterministic", "mcdo", "edl"]
base_dir = "./"

results_summary = {}

for task in tasks:
    results_summary[task] = {}
    for method in methods:
        folder = f"{task}/results/isic2018/{method}"
        # Collect all metric JSON files
        metrics_list = []
        for run_folder in os.listdir(folder):
            metrics_path = os.path.join(folder, run_folder, "eval", f"metrics_{method}.json")
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                metrics_list.append(metrics)
        
        # Collect per-metric lists
        all_metrics = {}
        for metric in metrics_list[0].keys():
            all_metrics[metric] = [m[metric] for m in metrics_list]
        
        # Compute mean and std dev
        stats = {}
        for metric, values in all_metrics.items():
            stats[metric] = {
                "mean": round(float(np.mean(values)), 3),
                "std": round(float(np.std(values, ddof=1)), 3)  # sample std dev
            }
        
        results_summary[task][method] = stats

# Print nicely
import pprint
pprint.pprint(results_summary)
