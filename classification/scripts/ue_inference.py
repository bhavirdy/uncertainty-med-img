import argparse
import yaml

def run_mcdo_inference(config, model, dataloader):
    """Run MC Dropout inference"""
    pass

def run_deep_ensemble_inference(config, models, dataloader):
    """Run Deep Ensemble inference"""
    pass

def run_edl_inference(config, model, dataloader):
    """Run Evidential Deep Learning inference"""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UE Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # TODO: load model(s), dataset/dataloader based on config
    # TODO: select inference method (mcdo, deep ensemble, edl)
    # TODO: run inference and save predictions + uncertainties
